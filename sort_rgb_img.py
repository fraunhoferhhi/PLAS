# Load an image and sort it using the PLAS algorithm
# The output is saved in the same directory as the input image, with the suffix "_sorted.png" added to the filename.

from PIL import Image
from plas import sort_with_plas
import math
import torch
import torchvision
import click
import os

from vad import compute_vad

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

    vad_img = compute_vad(tensor_image.permute(1, 2, 0).cpu().numpy() * 255)
    print(f"VAD of image: {vad_img:.4f}")

    if shuffle:
        # shuffle the image to avoid local minimum
        tensor_image = tensor_image.flatten(start_dim=1)[:, torch.randperm(tensor_image.shape[1] * tensor_image.shape[2])].reshape(tensor_image.shape)

        # TODO: see above for non-square images
        vad_img_shuf = compute_vad(tensor_image.permute(1, 2, 0).cpu().numpy() * 255)
        print(f"VAD of shuffled image: {vad_img_shuf:.4f}")


    sorted_img, grid_indices = sort_with_plas(tensor_image, improvement_break=1e-4, min_blur_radius=1,
                                              border_type_x="reflect", border_type_y="reflect", verbose=True)

    output_file_stem = os.path.join(os.path.dirname(img_path), os.path.basename(img_path).split(".")[0])
    if shuffle:
        output_file_stem += "_shuffled"

    torchvision.utils.save_image(sorted_img, f"{output_file_stem}.png")

    vad_sorted_img = compute_vad(sorted_img.permute(1, 2, 0).cpu().numpy() * 255)
    print(f"VAD of sorted image: {vad_sorted_img:.4f}")

    if not shuffle:
        # TODO restores in the wrong direction
        restored_img = tensor_image.flatten(start_dim=1)[:, grid_indices].squeeze(1)
        torchvision.utils.save_image(restored_img, f"{output_file_stem}_restored.png")

if __name__ == "__main__":
    sort_image()
