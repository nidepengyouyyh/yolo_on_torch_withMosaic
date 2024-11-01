import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from data_pre.formatting import PackDetInputs
from data_pre.loading import LoadImageFromFile, LoadAnnotations
from data_pre.resize import YOLOv5KeepRatioResize, LetterResize
from dataset import transform_labels

# Step 1: 定义类别名称到标签编号的映射字典
class_name_to_label = {
    "plane": 0,
    "ship": 1,
    "storage-tank": 2,
    "baseball-diamond": 3,
    "tennis-court": 4,
    "basketball-court": 5,
    "ground-track-field": 6,
    "harbor": 7,
    "bridge": 8,
    "small-vehicle": 9,
    "large-vehicle": 10,
    "helicopter": 11,
    "roundabout": 12,
    "soccer-ball-field": 13,
    "swimming-pool": 14,
}


class DOTA_dataset(Dataset):
    def __init__(self, image_dir, label_dir=None, transform=None, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.device = device
        self.samples = []
        image_files = os.listdir(image_dir)
        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(label_dir, label_file)
            if os.path.exists(label_path):
                self.samples.append((image_path, label_path))

        if transform:
            self.transforms = [
                LoadImageFromFile(),
                YOLOv5KeepRatioResize(scale=(640, 640)),
                LetterResize(scale=(640, 640), allow_scale_up=False, pad_val={'img': 114}),
                LoadAnnotations(with_bbox=True),
                PackDetInputs(meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'pad_param'))
            ]
        else:
            self.transforms = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path, label_path = self.samples[idx]
        img_data = {"img_path": image_path, "img_id": idx}

        # Step 2: 读取并转换标签
        with open(label_path, 'r') as f:
            lines = f.readlines()
            bboxes = []
            if len(lines) == 0:
                bboxes = torch.empty((0, 5), dtype=torch.float32)
            for line in lines:
                parts = line.strip().split()
                x1, y1, x2, y2, name, _ = parts[0:6]
                label = class_name_to_label.get(name, -1)  # 获取类别编号
                if label == -1:
                    print(f"Warning: '{name}' 类别未找到，跳过该标签")
                    continue
                bboxes.append([label, float(x1), float(y1), float(x2), float(y2)])

        bboxes = torch.tensor(bboxes, dtype=torch.float32).to(self.device).clone()

        # Step 3: 应用变换
        for t in self.transforms:
            results = t(img_data)

        img = results['inputs'].to(self.device)
        data_sample = results['data_samples'].metainfo
        bboxes = transform_labels(bboxes, data_sample['ori_shape'], data_sample['img_shape'])
        img_metas = {'batch_input_shape':img.shape[-2:]}
        data = {'input':img,
                'datasamples':{
                    'bboxes_labels': bboxes,
                    'img_metas': img_metas
                }}

        return data