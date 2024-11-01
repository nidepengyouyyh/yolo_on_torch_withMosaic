import torch
from typing import List
from models.structures import InstanceData

# 类别到ID的映射字典，假设 small-vehicle 是类别 0
CLASS_NAME_TO_ID = {
    'small-vehicle': 0,
    # 这里可以根据你的数据集增加更多类别
}

def load_txt_to_instance_data(txt_file_path: str) -> InstanceData:
    """
    从标签txt文件读取数据，并转换为InstanceData对象。

    Args:
        txt_file_path (str): txt文件的路径，每一行包含一个目标的信息，
                             格式为：<x_min> <y_min> <x_max> <y_max> <class_name> <ignore_flag>

    Returns:
        InstanceData: 包含bboxes和labels的InstanceData对象。
    """
    bboxes = []
    labels = []

    # 读取txt文件
    with open(txt_file_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            x_min = float(parts[0])
            y_min = float(parts[1])
            x_max = float(parts[2])
            y_max = float(parts[3])
            class_name = parts[4]
            ignore_flag = int(parts[5])

            # 如果 ignore_flag 为 1，跳过此目标
            if ignore_flag == 1:
                continue

            # 获取类别ID
            class_id = CLASS_NAME_TO_ID.get(class_name, -1)
            if class_id == -1:
                print(f"警告: 未知类别 {class_name}，跳过此目标。")
                continue

            # 添加到列表
            labels.append(class_id)
            bboxes.append([x_min, y_min, x_max, y_max])

    # 将列表转换为Tensor
    bboxes = torch.tensor(bboxes, dtype=torch.float32)  # 形状为[N, 4]
    labels = torch.tensor(labels, dtype=torch.int64)  # 形状为[N]

    # 创建InstanceData对象
    instance_data = InstanceData(bboxes=bboxes, labels=labels)

    return instance_data

def convert_txts_to_batch_gt_instances(txt_file_paths: List[str]) -> List[InstanceData]:
    """
    将多个txt文件转换为batch_gt_instances格式。

    Args:
        txt_file_paths (List[str]): txt文件路径列表，每个路径对应一张图像的标签。

    Returns:
        List[InstanceData]: 包含多个InstanceData对象的列表，每个对象对应一张图像的标签。
    """
    batch_gt_instances = [
        load_txt_to_instance_data(txt_file_path) for txt_file_path in txt_file_paths
    ]
    return batch_gt_instances

def parse_bbox_from_file(file_path):
    """
    读取文件中的边界框信息，并将其转换为张量。

    Args:
        file_path (str): 文件路径，文件内容形如
                         "x_min y_min x_max y_max class_name"

    Returns:
        Tensor: 形状为 (num_instances, 6)，每行数据代表 [image_index, class_id, x_min, y_min, x_max, y_max]
    """
    # 用于存储所有的边界框信息
    bbox_list = []

    # 读取文件并解析每一行
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue  # 跳过不完整的数据行

            # 解析边界框的坐标
            _, class_id, x_min, y_min, x_max, y_max = map(float, parts[:6])
            # class_id = 0  # 可以根据需要设置类别ID，这里默认设置为0

            # 假设image_index为0，因为没有提供此信息
            image_index = 0

            # 将数据存入列表，顺序为 [image_index, class_id, x_min, y_min, x_max, y_max]
            bbox_list.append([image_index, class_id, x_min, y_min, x_max, y_max])

    # 转换为张量
    bbox_tensor = torch.tensor(bbox_list, dtype=torch.float32)
    return bbox_tensor