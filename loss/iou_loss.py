import torch
import torch.nn as nn
from typing import Optional, Union, Tuple
import math

def bbox_overlaps(pred: torch.Tensor,
                  target: torch.Tensor,
                  iou_mode: str = 'ciou',
                  bbox_format: str = 'xywh',
                  siou_theta: float = 4.0,
                  eps: float = 1e-7) -> torch.Tensor:
    assert iou_mode in ('iou', 'ciou', 'giou', 'siou')
    assert bbox_format in ('xyxy', 'xywh')

    if bbox_format == 'xywh':
        # Convert from (x, y, w, h) to (x1, y1, x2, y2)
        pred = torch.cat((pred[..., :2] - pred[..., 2:] / 2,
                          pred[..., :2] + pred[..., 2:] / 2), dim=-1)
        target = torch.cat((target[..., :2] - target[..., 2:] / 2,
                            target[..., :2] + target[..., 2:] / 2), dim=-1)

    # Extract coordinates
    bbox1_x1, bbox1_y1 = pred[..., 0], pred[..., 1]
    bbox1_x2, bbox1_y2 = pred[..., 2], pred[..., 3]
    bbox2_x1, bbox2_y1 = target[..., 0], target[..., 1]
    bbox2_x2, bbox2_y2 = target[..., 2], target[..., 3]

    # Calculate overlap
    overlap = (torch.min(bbox1_x2, bbox2_x2) - torch.max(bbox1_x1, bbox2_x1)).clamp(min=0) * \
              (torch.min(bbox1_y2, bbox2_y2) - torch.max(bbox1_y1, bbox2_y1)).clamp(min=0)

    # Calculate union
    w1, h1 = bbox1_x2 - bbox1_x1, bbox1_y2 - bbox1_y1
    w2, h2 = bbox2_x2 - bbox2_x1, bbox2_y2 - bbox2_y1
    union = (w1 * h1) + (w2 * h2) - overlap + eps

    # Calculate IoU
    ious = overlap / union

    # Enclosing area for CIoU/GIoU
    enclose_x1y1 = torch.min(pred[..., :2], target[..., :2])
    enclose_x2y2 = torch.max(pred[..., 2:], target[..., 2:])
    enclose_wh = (enclose_x2y2 - enclose_x1y1).clamp(min=0)
    enclose_w = enclose_wh[..., 0]
    enclose_h = enclose_wh[..., 1]

    if iou_mode == 'ciou':
        enclose_area = enclose_w**2 + enclose_h**2 + eps
        rho2 = ((bbox2_x1 + bbox2_x2 - bbox1_x1 - bbox1_x2)**2 +
                (bbox2_y1 + bbox2_y2 - bbox1_y1 - bbox1_y2)**2) / 16

        v = (4 / (math.pi**2)) * torch.pow(
            torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        with torch.no_grad():
            alpha = v / (v - ious + (1 + eps))

        ious = ious - (rho2 / enclose_area + alpha * v)

    elif iou_mode == 'giou':
        convex_area = enclose_w * enclose_h + eps
        ious = ious - (convex_area - union) / convex_area

    elif iou_mode == 'siou':
        sigma_cw = (bbox2_x1 + bbox2_x2) / 2 - (bbox1_x1 + bbox1_x2) / 2 + eps
        sigma_ch = (bbox2_y1 + bbox2_y2) / 2 - (bbox1_y1 + bbox1_y2) / 2 + eps
        sigma = torch.sqrt(sigma_cw**2 + sigma_ch**2)

        sin_alpha = torch.abs(sigma_ch) / sigma
        sin_alpha = torch.where(sin_alpha <= math.sin(math.pi / 4), sin_alpha,
                                torch.abs(sigma_cw) / sigma)
        angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)

        rho_x = (sigma_cw / enclose_w)**2
        rho_y = (sigma_ch / enclose_h)**2
        gamma = 2 - angle_cost
        distance_cost = (1 - torch.exp(-gamma * rho_x)) + \
                        (1 - torch.exp(-gamma * rho_y))

        omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
        omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
        shape_cost = (1 - torch.exp(-omiga_w))**siou_theta + \
                     (1 - torch.exp(-omiga_h))**siou_theta

        ious = ious - (distance_cost + shape_cost) * 0.5

    return ious.clamp(min=-1.0, max=1.0)

def reduce_loss(loss: torch.Tensor, reduction: str = 'mean') -> torch.Tensor:
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    elif reduction == 'none':
        return loss
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")

def weight_reduce_loss(loss: torch.Tensor,
                       weight: Optional[torch.Tensor] = None,
                       reduction: str = 'mean',
                       avg_factor: Optional[float] = None) -> torch.Tensor:
    if weight is not None:
        loss = loss * weight

    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        if reduction == 'mean':
            eps = torch.finfo(loss.dtype).eps
            loss = loss.sum() / (avg_factor + eps)
        elif reduction != 'none':
            raise ValueError('avg_factor cannot be used with reduction="sum"')
    return loss

class IoULoss(nn.Module):
    """IoULoss.

    Compute the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        iou_mode (str): Options are "ciou", "siou", "giou".
            Defaults to "ciou".
        bbox_format (str): Options are "xywh" and "xyxy".
            Defaults to "xywh".
        eps (float): Eps to avoid log(0). Defaults to 1e-7.
        reduction (str): Options are "none", "mean" and "sum". Defaults to "mean".
        loss_weight (float): Weight of the loss. Defaults to 1.0.
        return_iou (bool): If True, returns loss and IoU. Defaults to True.
    """

    def __init__(self,
                 iou_mode: str = 'ciou',
                 bbox_format: str = 'xywh',
                 eps: float = 1e-7,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0,
                 return_iou: bool = True):
        super(IoULoss, self).__init__()
        assert bbox_format in ('xywh', 'xyxy'), \
            "bbox_format should be either 'xywh' or 'xyxy'."
        assert iou_mode in ('ciou', 'siou', 'giou'), \
            "iou_mode should be one of 'ciou', 'siou', 'giou'."
        self.iou_mode = iou_mode
        self.bbox_format = bbox_format
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.return_iou = return_iou

    def forward(self,
                pred: torch.Tensor,
                target: torch.Tensor,
                weight: Optional[torch.Tensor] = None,
                avg_factor: Optional[float] = None,
                reduction_override: Optional[Union[str, bool]] = None
                ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], torch.Tensor]:
        """Forward function.

        Args:
            pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2)
                or (x, y, w, h), shape (n, 4).
            target (Tensor): Corresponding ground truth bboxes, shape (n, 4).
            weight (Tensor, optional): Element-wise weights.
            avg_factor (float, optional): Average factor when computing the mean of losses.
            reduction_override (str, bool, optional): Same as built-in losses of PyTorch.
                Defaults to None.
        Returns:
            loss or tuple(loss, iou): IoU loss and optionally IoU value.
        """
        if weight is not None and not torch.any(weight > 0):
            if pred.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred * weight).sum()  # Returns 0 if all weights are zero

        # Override reduction method if specified.
        reduction = reduction_override if reduction_override else self.reduction

        if weight is not None and weight.dim() > 1:
            weight = weight.mean(-1)

        # Calculate IoU between predicted and target boxes.
        iou = bbox_overlaps(
            pred,
            target,
            iou_mode=self.iou_mode,
            bbox_format=self.bbox_format,
            eps=self.eps
        )

        # Calculate the loss as 1 - IoU.
        loss = self.loss_weight * weight_reduce_loss(1.0 - iou, weight, reduction, avg_factor)

        # Return both loss and IoU if specified.
        if self.return_iou:
            return loss, iou
        else:
            return loss