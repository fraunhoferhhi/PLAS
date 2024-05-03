# Load an image and sort it using the PLAS algorithm
# The output is saved in the same directory as the input image, with the suffix "_sorted.png" added to the filename.

from PIL import Image
from plas import sort_with_plas
import math
import torch
import torchvision
import click
import os

@click.command()
@click.option("--img-path", type=click.Path(exists=True))
@click.option("--shuffle/--no-shuffle", default=True)
def sort_image(img_path, shuffle):

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = "cpu"

    print(f"Using device: {device}")

    # Read an image and put it on the GPU
    image = Image.open(img_path).convert("RGB")
    tensor_image = torchvision.transforms.ToTensor()(image).to(device)

    # Currently only support square images, so
    sidelen = int(math.sqrt(tensor_image.shape[1] * tensor_image.shape[2]))

    # truncate the image
    img_trunc = tensor_image.flatten(start_dim=1)[:, :sidelen * sidelen]

    if shuffle:
        # shuffle the image to avoid local minimum
        img_trunc = img_trunc[:, torch.randperm(sidelen * sidelen)]

    # reshape the image to be a square
    img_trunc_shuf_sq = img_trunc.reshape(-1, sidelen, sidelen)

    sorted_img, grid_indices = sort_with_plas(img_trunc_shuf_sq, improvement_break=1e-4, border_type_x="reflect", border_type_y="reflect", verbose=True)

    output_file = os.path.basename(img_path).split(".")[0]
    if shuffle:
        output_file += "_shuffled"
    output_file += "_sorted.png"

    torchvision.utils.save_image(sorted_img, os.path.join(os.path.dirname(img_path), output_file))

if __name__ == "__main__":
    sort_image()