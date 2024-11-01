import os

from data_pre.data_preprocess import preprocess_image
from data_pre.formatting import PackDetInputs
from data_pre.loading import LoadImageFromFile, LoadAnnotations
from data_pre.resize import YOLOv5KeepRatioResize, LetterResize
from data_pre.tolabel import parse_bbox_from_file
from loss.parse_loss import parse_losses
from models.basemodule import BaseModule
from models.csp_darknet import YOLOv8CSPDarknet
from models.optim import YOLOv5OptimizerConstructor
from models.yolov8_pafpn import YOLOv8PAFPN
from models.yolov8_head import YOLOv8Head
from utils import stack_batch
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_image_with_boxes(img_tensor, labels_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    # 移除批量维度
    img = img_tensor.squeeze(0)

    # 反归一化
    img = img * torch.tensor(std).view(-1, 1, 1) + torch.tensor(mean).view(-1, 1, 1)

    # 将值缩放到[0, 255]并转换为uint8
    img = (img * 255).clamp(0, 255).byte()

    # 转换为numpy格式以便在Matplotlib中显示
    img_np = img.permute(1, 2, 0).cpu().numpy()

    # 显示图像并添加边框
    fig, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(img_np)

    # 绘制标签框
    for label in labels_tensor:
        _, _, x1, y1, x2, y2 = label  # 获取标签框的坐标
        width, height = x2 - x1, y2 - y1
        rect = patches.Rectangle((x1, y1), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.axis('off')
    plt.show()

class Detector(BaseModule):
    def __init__(self):
        super().__init__(None)
        self.backbone = YOLOv8CSPDarknet(deepen_factor=1.0, widen_factor=1.0, last_stage_out_channels=512)
        self.neck = YOLOv8PAFPN([256, 512, 512], [256, 512, 512], deepen_factor=1.0, widen_factor=1.0)
        self.bbox_head = YOLOv8Head(num_classes=80, in_channels=[256, 512, 512],
                                    train_cfg={'assigner': {'type': 'BatchTaskAlignedAssigner', 'num_classes': 80, 'use_ciou': True, 'topk': 10, 'alpha': 0.5, 'beta': 6.0, 'eps': 1e-09}})

    def set_train(self):
        optim_wrapper_config = {'type': 'OptimWrapper', 'clip_grad': {'max_norm': 10.0}, 'optimizer': {'type': 'SGD', 'lr': 0.1, 'momentum': 0.937, 'weight_decay': 0.0005, 'nesterov': True, 'batch_size_per_gpu': 1}}

        self.optim_wrapper = YOLOv5OptimizerConstructor(optim_wrapper_cfg=optim_wrapper_config,
                                                        paramwise_cfg=None)(self)

    def forward(self, x, data_sample):
        x = preprocess_image(x, mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], bgr_to_rgb=True)
        x = self.backbone(x)
        x = self.neck(x)
        losses = self.bbox_head.loss(x, data_sample)
        parsed_losses, log_vars = parse_losses(losses)
        self.optim_wrapper.update_params(parsed_losses)
        return log_vars

def load_weights_with_mapping(model, weight_path):
    # 加载权重文件
    checkpoint = torch.load(weight_path, map_location='cpu')
    model_weights = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint

    # # 创建一个新的字典用于存储映射后的权重
    # new_state_dict = {}
    # for model_key in models.state_dict().keys():
    #     # 获取权重文件中对应的键名
    #     # checkpoint_key = model_key.replace('stem.', 'backbone.stem.')
    #
    #     if model_key in model_weights:
    #         new_state_dict[model_key] = model_weights[model_key]
    #     else:
    #         print(f"{model_key}: Not found in weight file.")

    # 加载映射后的权重
    model.load_state_dict(model_weights, strict=False)
    return model

def transform_labels(labels_tensor, ori_shape, img_shape):
    # Unpack original and target dimensions
    ori_h, ori_w = ori_shape
    img_h, img_w = img_shape[:2]

    # 1. Calculate scale factors for aspect-ratio-preserving resize
    scale_x = img_w / ori_w
    scale_y = img_h / ori_h
    scale = min(scale_x, scale_y)  # Choose the smallest to keep aspect ratio

    # 2. Apply the scaling to all bounding box coordinates
    labels_tensor[:, 2] *= scale  # x1
    labels_tensor[:, 3] *= scale  # y1
    labels_tensor[:, 4] *= scale  # x2
    labels_tensor[:, 5] *= scale  # y2

    # 3. Calculate padding if the aspect ratio doesn't match
    pad_x = (img_w - ori_w * scale) / 2
    pad_y = (img_h - ori_h * scale) / 2

    # 4. Adjust for padding by adding offsets
    labels_tensor[:, 2] += pad_x  # x1
    labels_tensor[:, 3] += pad_y  # y1
    labels_tensor[:, 4] += pad_x  # x2
    labels_tensor[:, 5] += pad_y  # y2

    return labels_tensor

# Example usage:
# labels_tensor = torch.tensor([[0, 0, x1, y1, x2, y2], ...])
# adjusted_labels = transform_labels(labels_tensor)



def train(model, data_id):




    # 指定输入图片的路径
    image_path = f'./data/coco/val2017/{data_id}.jpg'  # 替换为你的图片路径
    annfile_path = f'./data/coco/annfiles/{data_id}.txt'
    image = cv2.imread(image_path)

    transforms = []
    transforms.append(LoadImageFromFile())
    transforms.append(YOLOv5KeepRatioResize(scale=(640, 640)))
    transforms.append(LetterResize(scale=(640, 640), allow_scale_up=False, pad_val={'img': 114}))
    transforms.append(LoadAnnotations(with_bbox=True))
    transforms.append(PackDetInputs(meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param')))
    results = {"img_path": image_path, "img_id": 0}
    for t in transforms:
        results = t(results)
    batch_input = [results['inputs'].float()]
    data_ = results
    data_['inputs'] = stack_batch(batch_input, 1, 0)
    data_['data_samples'] = [data_['data_samples']]

    data_samples = {}
    data_samples['img_metas'] = [
        data_samples.metainfo for data_samples in data_['data_samples']
    ]
    instance_data = parse_bbox_from_file(annfile_path)
    instance_data = transform_labels(instance_data, data_samples['img_metas'][0]['ori_shape'], data_samples['img_metas'][0]['img_shape'])
    data_samples['bboxes_labels'] = instance_data
    # show_image_with_boxes(data_['inputs'], instance_data)


    results = model(data_['inputs'], data_samples)
    print(results)

def get_image_names(folder_path):
    # 初始化一个空列表用于保存图像名称
    image_names = []

    # 遍历指定文件夹下的所有文件
    for filename in os.listdir(folder_path):
        # 检查文件是否以 .jpg 结尾
        if filename.endswith('.jpg'):
            # 去掉扩展名并添加到列表
            image_name = os.path.splitext(filename)[0]
            image_names.append(image_name)

    return image_names
if __name__ == '__main__':
    model = Detector()
    model.init_weights()
    weight_path = './checkpoint/yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth'
    # models = load_weights_with_mapping(models, weight_path=weight_path)

    model.set_train()
    for i in get_image_names('./data/coco/val2017'):
        try:
            train(model, i)
        except:
            print('not found')
        # cv2.waitKey(0)
        if i == 3:
            break

