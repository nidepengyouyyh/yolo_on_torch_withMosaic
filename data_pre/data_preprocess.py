import torch
def preprocess_image(image: torch.Tensor, mean: list, std: list, bgr_to_rgb: bool) -> torch.Tensor:
    """
    Preprocess the input image tensor using mean, std normalization and optional BGR to RGB conversion.

    Args:
        image (torch.Tensor): Input image tensor of shape [batch_size, 3, 640, 640].
        mean (list): Mean values for each channel for normalization.
        std (list): Standard deviation values for each channel for normalization.
        bgr_to_rgb (bool): Whether to convert from BGR to RGB.

    Returns:
        torch.Tensor: Preprocessed image tensor of shape [batch_size, 3, 640, 640].
    """
    # Convert the image from BGR to RGB if specified
    if bgr_to_rgb:
        image = image[:, [2, 1, 0], :, :]  # Swap channel order from BGR to RGB

    # Normalize the image: (image - mean) / std
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(image.device)
    std = torch.tensor(std).view(1, 3, 1, 1).to(image.device)
    image = (image - mean) / std

    return image