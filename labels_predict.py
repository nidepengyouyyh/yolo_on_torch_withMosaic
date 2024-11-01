import os
import torch
from pathlib import Path
from tqdm import tqdm
import numpy as np

from load import load_weights_with_mapping
from models.model import Detector
from test import pretrain_test_model


def get_image_paths(folder_path):
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']  # 图片扩展名列表
    image_paths = []  # 存储找到的图片路径

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                full_path = os.path.join(root, file)
                image_paths.append(full_path)

    return image_paths

def get_label(label_path, image_name):
    file_path = f"{label_path}/{image_name}.txt"
    labels = []

    try:
        with open(file_path, "r") as file:
            labels = [line.strip().split()[:5] for line in file.readlines()]
    except FileNotFoundError:
        print(f"文件未找到: {file_path}")

    return labels


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def calculate_iou(box1, box2):
    x1, y1, x2, y2 = map(float, box1)
    x1b, y1b, x2b, y2b = map(float, box2)
    int_x1 = max(x1, x1b)
    int_y1 = max(y1, y1b)
    int_x2 = min(x2, x2b)
    int_y2 = min(y2, y2b)

    inter_area = max(0, int_x2 - int_x1) * max(0, int_y2 - int_y1)
    union_area = (x2 - x1) * (y2 - y1) + (x2b - x1b) * (y2b - y1b) - inter_area

    return inter_area / union_area if union_area > 0 else 0

def compute_map(detections, ground_truths, iou_threshold=0.5):
    all_class_ids = set()
    for gts in ground_truths:
        for gt in gts:
            class_id = gt[4]
            all_class_ids.add(class_id)
    num_classes = len(all_class_ids)
    all_class_ids = list(all_class_ids)
    aps = []
    detections = [[item.split(' ') for item in sublist] for sublist in detections]
    for class_id in range(num_classes):
        scores = []
        true_positives = []
        false_positives = []

        for img_idx in range(len(detections)):

            det_boxes = [det[:4] for det in detections[img_idx] if det[4] == all_class_ids[class_id]]
            det_scores = [det[5] for det in detections[img_idx] if det[4] == all_class_ids[class_id]]

            gt_boxes = [gt[:4] for gt in ground_truths[img_idx] if gt[4] == all_class_ids[class_id]]

            # Sort detections by score
            sorted_indices = np.argsort(det_scores)[::-1]
            det_boxes = [det_boxes[i] for i in sorted_indices]
            det_scores = [det_scores[i] for i in sorted_indices]

            for det_box, det_score in zip(det_boxes, det_scores):
                scores.append(det_score)
                best_iou = 0
                for gt_box in gt_boxes:
                    iou = calculate_iou(det_box, gt_box)
                    if iou > best_iou:
                        best_iou = iou

                if best_iou >= iou_threshold:
                    true_positives.append(1)
                    false_positives.append(0)
                    # Remove matched GT box
                    gt_boxes = [gt for gt in gt_boxes if calculate_iou(gt, det_box) < iou_threshold]
                else:
                    true_positives.append(0)
                    false_positives.append(1)

        # Compute precision-recall curve
        if len(scores) > 0:
            sort_indices = np.argsort(scores)[::-1]
            tp_sorted = np.array(true_positives)[sort_indices].cumsum()
            fp_sorted = np.array(false_positives)[sort_indices].cumsum()
            precision = tp_sorted / (tp_sorted + fp_sorted)
            all_labels = []
            for img_gt in ground_truths:
                # 遍历当前图像中的所有标签
                for label in img_gt:
                    if label[4] == all_class_ids[class_id]:
                        all_labels.append(label[4])
            total_true_positives = sum([1 for label in all_labels if all_class_ids[class_id]])
            # 计算召回率
            recall = tp_sorted / total_true_positives
            ap = compute_ap(recall, precision)
            aps.append(ap)

    return np.mean(aps)


def compare_labels(predicted_labels, real_labels, iou_threshold=0.5):
    # 初始化统计数据
    true_positives = []
    false_positives = []
    false_negatives = len(real_labels)  # 初始假阴性为真实标签的数量

    # 复制真实标签列表，以便后续移除已匹配的标签
    real_labels_copy = real_labels[:]

    # 遍历每个预测框
    for pred in predicted_labels:
        if isinstance(pred, str):
            pred = pred.split(' ')
        det_box = [float(p) for p in pred[:4]]
        det_score = float(pred[5])
        det_label = int(pred[4])

        match_found = False
        for idx, real_label in enumerate(real_labels_copy):
            gt_box = [float(p) for p in real_label[:4]]
            gt_label = int(real_label[4])

            if det_label == gt_label:
                iou = calculate_iou(det_box, gt_box)
                if iou >= iou_threshold:
                    true_positives.append(1)
                    false_positives.append(0)
                    match_found = True
                    # 移除已匹配的真实框
                    del real_labels_copy[idx]
                    break

        if not match_found:
            true_positives.append(0)
            false_positives.append(1)

    # 统计剩余未被匹配的真实框作为假阴性
    false_negatives = len(real_labels_copy)

    return true_positives, false_positives, false_negatives


def labels_predict(dic_path, pretrain_path, label_path, save_path, device, save_labels = False):
    image_paths = get_image_paths(dic_path)[:10] #reduce batch
    yolo_model = Detector().to(device)
    yolo_model = load_weights_with_mapping(yolo_model, pretrain_path)
    os.makedirs(save_path, exist_ok=True)

    all_true_positives = []
    all_false_positives = []
    ground_truths = []
    predictions = []

    for image_path in tqdm(image_paths, desc="Processing images", unit="image"):
        image_name = Path(image_path).stem
        output_file = os.path.join(save_path, f"{image_name}.txt")
        pre_labels = pretrain_test_model(image_path, yolo_model, device)

        # 获取真实标签
        real_labels = get_label(label_path, image_name)

        # 输出预测标签与真实标签进行比较的信息
        true_positives, false_positives, false_negatives = compare_labels(pre_labels, real_labels)

        # 更新全局统计数据
        all_true_positives.extend(true_positives)
        all_false_positives.extend(false_positives)

        # 收集预测结果
        predictions.append(pre_labels)
        ground_truths.append(real_labels)

        if save_labels == True:
            with open(output_file, 'w') as f:
                for label in pre_labels:
                    f.write(label)

    # 计算mAP
    mAP = compute_map(predictions, ground_truths)
    print(f"mAP: {mAP}")


dic_path = r"F:\study\CODEs\xiaotiao\coco\val2017"
pre_model_path = "yolov8_l_syncbn_fast_8xb16-500e_coco_20230217_182526-189611b6.pth"
save_path = "save_label"
label_path = "F:/study/CODEs/xiaotiao/coco/annfiles"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
labels_predict(dic_path,pre_model_path, label_path, save_path, device)