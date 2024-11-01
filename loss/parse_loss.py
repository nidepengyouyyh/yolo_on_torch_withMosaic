import torch
from collections import OrderedDict
from typing import Dict, Tuple

def parse_losses(losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Parses raw output losses from the network.

    Args:
        losses (dict): Raw output of the network, usually containing
                       various loss components as tensors.

    Returns:
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]: A tuple where the first element
        is the total loss tensor, used for backpropagation, and the second element
        is an ordered dictionary of each loss component, useful for logging.
    """
    log_vars = OrderedDict()
    total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)

    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            mean_loss = loss_value.mean()
            log_vars[loss_name] = mean_loss
            if 'loss' in loss_name:  # We assume keys with 'loss' contribute to the total loss
                total_loss += mean_loss
        elif isinstance(loss_value, list) and all(isinstance(v, torch.Tensor) for v in loss_value):
            mean_loss = sum(v.mean() for v in loss_value)
            log_vars[loss_name] = mean_loss
            if 'loss' in loss_name:
                total_loss += mean_loss
        else:
            raise TypeError(f"Expected tensor or list of tensors for {loss_name}, but got {type(loss_value)}")

    log_vars['total_loss'] = total_loss  # Add total loss to the logs for completeness
    return total_loss, log_vars