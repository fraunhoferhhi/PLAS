import numpy as np
import torch

from PIL import Image
import io

def tensor_to_png(tensor):
    """
    Convert a PyTorch tensor to a PNG image in memory.

    Args:
    tensor (torch.Tensor): The input tensor with shape (C, H, W) where C must be 1, 3 or 4.

    Returns:
    bytes: The PNG image data.
    """
    # Ensure the tensor is on CPU and convert to numpy
    tensor = tensor.cpu().numpy()

    # Normalize the tensor to the range [0, 255] for PNG encoding
    tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min()) * 255
    tensor = tensor.astype(np.uint8)

    # Convert to H x W x C format for PIL
    if tensor.shape[0] == 1:  # Grayscale
        tensor = tensor.squeeze(0)
    elif tensor.shape[0] == 3 or tensor.shape[0] == 4:  # RGB or RGBA
        tensor = np.transpose(tensor, (1, 2, 0))
    else:
        raise ValueError("Expected tensor with either 1, 3 or 4 channels.")

    # Create a PIL image
    image = Image.fromarray(tensor)

    # Save the image to a bytes buffer
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def avg_L2_dist_between_neighbors(image):
    """Computes the average L2 distance between neighboring pixels. Two pixels are neighbored if they share a common edge.

    Args:
        image (np.ndarray or torch.Tensor): Must be of shape H x W or C x H x W where C is the number of channels, H the height and W the width.

    Raises:
        TypeError: raised if image neither an np.ndarray nor a torch.Tensor

    Returns:
        float: the desired average...
    """

    if isinstance(image, np.ndarray):
        if len(image.shape) == 2: 
            image = np.expand_dims(image, axis=0)
        elif len(image.shape) != 3:
            raise ValueError("Expected image with either 2 or 3 dimensions.")

        x_diff = np.diff(image, axis=2)
        x_L2_diff_sum = np.sum(np.sqrt(np.sum(x_diff * x_diff, axis=0)))
        y_diff = np.diff(image, axis=1)
        y_L2_diff_sum = np.sum(np.sqrt(np.sum(y_diff * y_diff, axis=0)))
    elif isinstance(image, torch.Tensor):
        if len(image.shape) == 2: 
            image = torch.unsqueeze(image, dim=0)
        elif len(image.shape) != 3:
            raise ValueError("Expected image with either 2 or 3 dimensions.")

        x_diff = torch.diff(image, axis=2)
        x_L2_diff_sum = torch.sum(torch.sqrt(torch.sum(x_diff * x_diff, axis=0)))
        y_diff = torch.diff(image, axis=1)
        y_L2_diff_sum = torch.sum(torch.sqrt(torch.sum(y_diff * y_diff, axis=0)))
    else:
        raise TypeError("Input must be a numpy array or a torch tensor")

    H, W = image.shape[1:]
    return ((x_L2_diff_sum + y_L2_diff_sum) / (H * (W - 1) + W * (H - 1))).item()
