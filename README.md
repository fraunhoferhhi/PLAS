# Parallel Linear Assignment Sorting (PLAS)
Sorting multi-dimensional data into locally smooth 2D grids.

The algorithm will sort tensors of shape *(C, n, n)*. It reorders the elements along the last two dimensions (grid columns and rows) with the attempt to minimize the $L^2$-distance between neighboring elements in $x$ and $y$ direction.

This method was developed for [Compact 3D Scene Representation via Self-Organizing Gaussian Grids](https://github.com/fraunhoferhhi/Self-Organizing-Gaussians).


### Example

We can use PLAS to sort the pixels of an image, where *C=3* from the three color channels of the image. An example implementation can be found in `sort_rgb_img.py`.

*Starry Night* by Vincent van Gogh             |  *Sorted Night* with PLAS
:-------------------------:|:-------------------------:
![A reproduction of the painting Starry Night by Vincent van Gogh](/img/VanGogh-starry_night.jpg)  | ![All pixels of the painting sorted with the algorithm](/img/VanGogh-starry_night_sorted.png)

The sorted output contains the same pixels as the input image, reordered to have high similarity between neighbors.


### Install

Create a virtual environment with Python 3.10 and PyTorch. Then 

```bash
pip install git+https://github.com/fraunhoferhhi/PLAS.git
```

### Usage

```Python
import plas

plas.sort_with_plas(...)
```

Have a look at the `examples/` folder for parameters and return values.


### Preparing data

Currently only square grids are supported in the implementation. Data needs to be truncated and reshaped into shape *(C, n, n)* before being passed into the sorting method.

If the input data is already partially ordered, you may consider shuffling it before calling PLAS, to avoid getting stuck in local minima. But given the iterative nature of the algorithm, a random shuffle will also increase the time to sort significantly.

