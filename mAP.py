import numpy as np
from pathlib import Path
from tqdm import tqdm


def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    """
    x1, y1, x2, y2 = box1
    x1b, y1b, x2b, y2b = box2
    x1, y1 = max(x1, x1b), max(y1, y1b)
    x2, y2 = min(x2, x2b), min(y2, y2b)

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    union_area = (x2 - x1b) * (y2b - y1b) + (x2b - x1) * (y2 - y1) - inter_area

    return inter_area / union_area if union_area > 0 else 0


def compute_ap(recall, precision):
    """
    Compute the average precision given recall and precision curves.
    """
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]

    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_map(detections, ground_truths, iou_threshold=0.5):
    """
    Compute the mean average precision (mAP) for object detection.
    """
    num_classes = len(set(label for gt in ground_truths for label in gt['labels']))
    aps = []

    for class_id in range(num_classes):
        scores = []
        true_positives = []
        false_positives = []

        for img_idx in range(len(detections)):
            det_boxes = [det[:4] for det in detections[img_idx] if det[4] == class_id]
            det_scores = [det[5] for det in detections[img_idx] if det[4] == class_id]

            gt_boxes = [box for box, label in zip(ground_truths[img_idx]['boxes'], ground_truths[img_idx]['labels']) if
                        label == class_id]

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
            recall = tp_sorted / sum([label == class_id for img_gt in ground_truths for label in img_gt['labels']])
            ap = compute_ap(recall, precision)
            aps.append(ap)

    return np.mean(aps)