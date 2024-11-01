import cv2
import torch
import torch.nn as nn

from data_pre.data_preprocess import preprocess_image
from data_pre.formatting import PackDetInputs
from data_pre.loading import LoadImageFromFile, LoadAnnotations
from data_pre.resize import YOLOv5KeepRatioResize, LetterResize
from models.basemodule import BaseModule
from models.csp_darknet import YOLOv8CSPDarknet
from models.yolov8_pafpn import YOLOv8PAFPN
from models.yolov8_head import YOLOv8Head
from utils import stack_batch, add_pred_to_datasample
import cv2

# 假设你有一个模型类
class Detector(BaseModule):
    def __init__(self):
        super().__init__(None)
        self.backbone = YOLOv8CSPDarknet(deepen_factor=1.0, widen_factor=1.0, last_stage_out_channels=512)
        self.neck = YOLOv8PAFPN([256, 512, 512], [256, 512, 512],deepen_factor=1.0, widen_factor=1.0)
        # head_cfg = {'multi_label': True, 'nms_pre': 30000, 'score_thr': 0.001,
        #             'nms': {'type': 'nms', 'iou_threshold': 0.7}, 'max_per_img': 300}
        self.bbox_head = YOLOv8Head(80, [256, 512, 512], test_cfg={'multi_label': True, 'nms_pre': 30000, 'score_thr': 0.001, 'nms': {'type': 'nms', 'iou_threshold': 0.7}, 'max_per_img': 300})

    def predict(self, x, data_sample):
        x = preprocess_image(x, mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0], bgr_to_rgb=True)
        x = self.backbone(x)
        x = self.neck(x)
        result = self.bbox_head.predict(x, data_sample)
        results = add_pred_to_datasample(data_sample, result)
        return results[0]


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


model = Detector()
ckp_path = 'yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth'
model = load_weights_with_mapping(model, ckp_path)

# 指定输入图片的路径
image_path = r'F:\study\CODEs\xiaotiao\coco\val2017\000000000802.jpg'  # 替换为你的图片路径
# image_path = r'mosaic/0.jpg'
image = cv2.imread(image_path)
transforms = []

transforms.append(LoadImageFromFile())
transforms.append(YOLOv5KeepRatioResize(scale=(640, 640)))
transforms.append(LetterResize(scale=(640, 640), allow_scale_up=False, pad_val={'img':114}))
transforms.append(LoadAnnotations(with_bbox=True))
transforms.append(PackDetInputs(meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param')))
results = {"img_path": image_path, "img_id":0}
for t in transforms:
    results = t(results)
batch_input = [results['inputs'].float()]
data_ = results
data_['inputs'] = stack_batch(batch_input, 1, 0)
data_['data_samples'] = [data_['data_samples']]

model.eval()
# 进行预测
results = model.predict(data_['inputs'], data_['data_samples'])
# result_list = []
# result_list.append(results)
results = results.cpu()
pred_instances = results.pred_instances
if 'scores' in pred_instances:
    pred_instances = pred_instances[
        pred_instances.scores > 0.1].cpu()  #设置阈值


boxes = pred_instances.bboxes  # tensor of shape (N, 4) with [x1, y1, x2, y2]
labels = pred_instances.labels  # tensor of shape (N,)
scores = pred_instances.scores

img = cv2.imread(image_path)

pred_instances = results.pred_instances[results.pred_instances.scores > 0.5]

for box, label, score in zip(boxes, labels, scores):
    # 提取坐标并将其转换为 int 类型
    x1, y1, x2, y2 = box.int().tolist()

    # 绘制检测框（蓝色框，厚度为2）
    cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

    # 构造标签文本，包括类别和置信度分数
    label_text = f'{int(label)}: {score:.2f}'
    # 在检测框的左上角写入标签文本（白色文字，字体大小0.5）
    cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), thickness=1)
cv2.imshow('Detections', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('good')
