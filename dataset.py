import os
import torch
from torch.utils.data import Dataset
from data_pre.formatting import PackDetInputs
from data_pre.loading import LoadImageFromFile, LoadAnnotations
from data_pre.resize import YOLOv5KeepRatioResize, LetterResize
import cv2
import random
import tempfile
import numpy as np
from data_augment.slide_windows import sliding_window_segmentation_in_memory
def transform_labels(labels_tensor, ori_shape, img_shape):
    # Unpack original and target dimensions
    ori_h, ori_w = ori_shape
    img_h, img_w = img_shape[:2]

    # 1. Calculate scale factors for aspect-ratio-preserving resize
    scale_x = img_w / ori_w
    scale_y = img_h / ori_h
    scale = min(scale_x, scale_y)  # Choose the smallest to keep aspect ratio

    # 2. Apply the scaling to all bounding box coordinates
    labels_tensor[:, 1] *= scale  # x1
    labels_tensor[:, 2] *= scale  # y1
    labels_tensor[:, 3] *= scale  # x2
    labels_tensor[:, 4] *= scale  # y2

    # 3. Calculate padding if the aspect ratio doesn't match
    pad_x = (img_w - ori_w * scale) / 2
    pad_y = (img_h - ori_h * scale) / 2

    # 4. Adjust for padding by adding offsets
    labels_tensor[:, 1] += pad_x  # x1
    labels_tensor[:, 2] += pad_y  # y1
    labels_tensor[:, 3] += pad_x  # x2
    labels_tensor[:, 4] += pad_y  # y2

    return labels_tensor

class Ydataset(Dataset):
    def __init__(self, image_dir, label_dir=None, transform=None, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        """
        Args:
            image_dir (string): 包含所有图片的目录路径。
            label_dir (string): 包含所有标签文件的目录路径。
            transform (callable, optional): 可选的transform函数，应用于图片上。
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = 640
        self.transform = transform
        self.samples = []
        self.device = device
        self.data_aug = 'mosaic'
        self.mosaic_border = (-640, -640)


        # 获取所有图片文件名
        image_files = os.listdir(image_dir)
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)
            if os.path.exists(label_path):
                self.samples.append((image_path, label_path))

        self.indices = list(range(len(self.samples)))
        if transform:
            self.transforms = []
            self.transforms.append(LoadImageFromFile())
            self.transforms.append(YOLOv5KeepRatioResize(scale=(640, 640)))
            self.transforms.append(LetterResize(scale=(640, 640), allow_scale_up=False, pad_val={'img': 114}))
            self.transforms.append(LoadAnnotations(with_bbox=True))
            self.transforms.append(PackDetInputs(meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param')))
        else:
            self.transforms = None


    def __len__(self):
        """返回数据集的大小"""
        return len(self.samples)

    def __getitem__(self, idx):
        """获取索引idx对应的样本"""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.data_aug == 'mosaic' and random.random() < 0.5:  # 50% 概率应用Mosaic增强
            img, bboxes, img_data = self.load_mosaic(idx)
        else:
            image_path, label_path = self.samples[idx]
            img_data = {"img_path": image_path, "img_id": idx}
            # 读取标签文件
            with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                bboxes = []
                for line in lines:
                    parts = line.strip().split()
                    x1, y1, x2, y2, label, num_id = map(float, parts)
                    bboxes.append([int(label), float(x1), float(y1), float(x2), float(y2)])
            bboxes = torch.tensor(bboxes, dtype=torch.float32).to(self.device)
            img = cv2.imread(image_path)

        for t in self.transforms:
            results = t(img_data)
        img = results['inputs'].to(self.device)
        data_sample = results['data_samples'].metainfo
        bboxes = transform_labels(bboxes, data_sample['ori_shape'], data_sample['img_shape'])
        img_metas = {'batch_input_shape': img.shape[-2:]}
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.tensor(bboxes, dtype=torch.float32).to(self.device)
        data = {
            'input': img,
            'datasamples': {
                'bboxes_labels': bboxes,
                'img_metas': img_metas
            }
        }


        return data

    def load_image(self, index):
        image_path, label_path = self.samples[index]
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        with open(label_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            bboxes = []
            for line in lines:
                parts = line.strip().split()
                x1, y1, x2, y2, label, num_id = map(float, parts)
                bboxes.append([int(label), float(x1), float(y1), float(x2), float(y2)])
        bboxes = np.array(bboxes, dtype=np.float32)  # 转换为NumPy数组
        img_data = {"img_path": image_path, "img_id": index}
        return img, bboxes, (h, w), img_data

    def load_mosaic(self, index):

        labels4 = []
        s = self.img_size
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.mosaic_border)  # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)  # 3 additional image indices
        random.shuffle(indices)
          # base image with 4 tiles
        img4 = np.full((s * 2, s * 2, 3), 114, dtype=np.uint8)

        first_img_name = indices[0]  # 需要实现 get_image_name 方法
        save_dir = 'mosaic'
        for i, idx in enumerate(indices):
            img, bboxes, (h, w), img_data = self.load_image(idx)
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
                # print('a')
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            if bboxes is not None and len(bboxes) > 0:
                bboxes = bboxes.copy()
                bboxes[:, 1:] += [padw, padh, padw, padh]  # 调整边界框位置
                labels4.extend(bboxes)


        if len(labels4) > 0:
            labels4 = np.array(labels4)
            np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])

        filename = f"{first_img_name}.jpg"  # 或者 .png，视需求而定
        save_path = os.path.join(save_dir, filename)
        cv2.imwrite(save_path, img4)

        # 创建 img_data 字典
        img_data = {
            "img_path": save_path,
            "img_id": index,
            "mosaic_center": (xc, yc),
            "indices_used": indices
        }

        # show_img = img4.copy()
        # for bbox in labels4:
        #     label, x1, y1, x2, y2 = bbox
        #     x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        #     cv2.rectangle(show_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        #     cv2.putText(show_img, str(int(label)), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        #
        # show_img = cv2.resize(show_img, (640, 640))
        # cv2.imshow('Mosaic Image with Labels', show_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return img4, labels4, img_data


def xywhn2xyxy(x, w, h, padw=0, padh=0):
    y = x.copy()
    y[:, 0] = w * (x[:, 0] - x[:, 2] / 2) + padw  # top left x
    y[:, 1] = h * (x[:, 1] - x[:, 3] / 2) + padh  # top left y
    y[:, 2] = w * (x[:, 0] + x[:, 2] / 2) + padw  # bottom right x
    y[:, 3] = h * (x[:, 1] + x[:, 3] / 2) + padh  # bottom right y
    return y

