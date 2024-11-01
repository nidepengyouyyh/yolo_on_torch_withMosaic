import cv2
import torch
from tqdm import tqdm

from dataset import Ydataset
from models.model import Detector
from transform import Transform
from torch.utils.data import DataLoader
from dataloader import custom_collate_fn
from load import load_weights_with_mapping
from utils import stack_batch
import matplotlib.pyplot as plt

def test_model(img_path, ckp_path, device):
    yolo_model = Detector().to(device)
    yolo_model = load_weights_with_mapping(yolo_model, ckp_path)
    transform = Transform()
    results = transform({'img_path': img_path, 'img_id': 0})
    batch_input = [results['inputs'].float().to(device)]
    data_ = results
    data_['inputs'] = stack_batch(batch_input, 1, 0)
    data_['data_samples'] = [data_['data_samples']]
    boxes, labels, scores = yolo_model.step_predict(data_['inputs'], data_['data_samples'])
    image = cv2.imread(img_path)
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
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


def pretrain_test_model(img_path, model, device):
    transform = Transform()
    results = transform({'img_path': img_path, 'img_id': 0})
    batch_input = [results['inputs'].float().to(device)]
    data_ = results
    data_['inputs'] = stack_batch(batch_input, 1, 0)
    data_['data_samples'] = [data_['data_samples']]
    boxes, labels, scores = model.step_predict(data_['inputs'], data_['data_samples'])
    image = cv2.imread(img_path)
    output_lines = []
    for box, label, score in zip(boxes, labels, scores):
        # 提取坐标并将其转换为 int 类型
        x1, y1, x2, y2 = box.int().tolist()
        # 绘制检测框（蓝色框，厚度为2）
        # print(x1, y1, x2, y2, label)
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

        # 构造标签文本，包括类别和置信度分数
        label_text = f'{int(label)}: {score:.2f}'
        # 在检测框的左上角写入标签文本（白色文字，字体大小0.5）
        cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), thickness=1)

        output_lines.append(f"{x1} {y1} {x2} {y2} {int(label)} {score:.2f}\n")
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')
    # plt.show()
    # cv2.waitKey(0)
    return output_lines





if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_dir = r'D:\study\CODEs\xiaotiao\coco\val2017\000000000139.jpg'
    ckp_dir = 'save_model/model_epoch_3_loss_60.4337.pth'
    yolo_model = Detector().to(device)
    yolo_model = load_weights_with_mapping(yolo_model, ckp_dir)
    # test_model(img_dir, ckp_dir, device)
    pretrain_test_model(img_dir, yolo_model, device)
