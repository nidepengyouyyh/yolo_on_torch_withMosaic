import torch

from dataset import Ydataset


def custom_collate_fn(batch):
    """
    自定义的 collate 函数，用于处理数据集中的每个批次。

    参数:
        batch (list): 每个元素可能是一个包含 'input' 和 'datasamples' 的字典，或者是一个这样的字典列表。

    返回:
        dict: 合并后的批次数据。
    """
    if isinstance(batch[0], list):
        # 如果 batch 中的每个元素是一个列表，需要先展平
        batch = [item for sublist in batch for item in sublist]

    inputs = torch.stack([item['input'] for item in batch])  # shape: [batchsize, 3, H, W]

    # 初始化 bbox 和 图像标签的列表
    bboxes_labels = []
    img_metas = []

    # 遍历 batch 中的每个样本，给每个 bbox 增加图像标签
    for batch_idx, item in enumerate(batch):
        bboxes = item['datasamples']['bboxes_labels']
        img_metas.append({'batch_input_shape': item['input'].shape[-2:]})
        # 在 bboxes 的第 0 维添加图像标签 (batch_idx)
        img_bboxes = torch.cat((torch.full((bboxes.shape[0], 1), batch_idx, device=bboxes.device), bboxes), dim=1)

        # 将带标签的 bboxes 添加到总列表中
        bboxes_labels.append(img_bboxes)

    # 将所有 bboxes 合并为一个 tensor，形状为 [m, 6]
    bboxes_labels = torch.cat(bboxes_labels, dim=0)

    # 构建 datasample 字典
    datasample = {
        'bboxes_labels': bboxes_labels,
        'img_metas': img_metas
    }
    return {'input': inputs, 'datasample': datasample}


# def custom_collate_slide():


# from torch.utils.data import DataLoader
#
# dataset = Ydataset(image_dir='path/to/images', label_dir='path/to/labels', transform=your_transform)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)