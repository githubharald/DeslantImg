# Deslanting Algorithm

* **Update 2021/2: added Python package**
* **Update 2021/1: added Python implementation**

This algorithm sets handwritten text in images upright, i.e. it removes the cursive writing style. One can use it as a
preprocessing step for handwritten text recognition. The following illustration shows input and output of the algorithm
for a given image (`data/test1.png`).

![deslanting](doc/example.png)

There are three implementations provided (Python, C++, OpenCL) with a focus on the Python implementation. For C++
and OpenCL see the folder [extras](extras).

## Installation

* Install by running `pip install .`
* Run `deslant_img` (without arguments) from the command line to process the images in the `data` directory (images
  taken from IAM and Bentham dataset)
* This opens a window showing the input image, deslanted image and score values
* The script can be configured via command line, see available options [below](#python-gui), or by
  running `deslant_img -h`

![plot](doc/plot.png)

## Usage

Command line options of `deslant_img`:

```
usage: deslant_img [-h] [--data DATA] [--optim_algo {grid,powell}]
               [--lower_bound LO] [--upper_bound HI]
               [--num_steps STEPS] [--bg_color BG]

optional arguments:
  -h, --help            show this help message and exit
  --data DATA           directory containing the (.png|.jpg|.bmp) input images
  --optim_algo {grid,powell}
                        either do grid search, or apply Powell's derivative-
                        free optimizer
  --lower_bound LO      lower bound of shear values
  --upper_bound HI      upper bound of shear values
  --num_steps STEPS     if grid search is used, this argument defines the
                        number if grid points
  --bg_color BG         color to fill the gaps of the sheared image that is
                        returned
```

## API

* Import the `deslant_img` function as shown in the code snippet
* For documentation of parameters see command line parameters above or use `help(deslant_img)`

````python
from deslant_img.__init__ import deslant_img
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('data/test1.png', cv2.IMREAD_GRAYSCALE)
res = deslant_img(img)

plt.imshow(res.img)
plt.show()
````

## Algorithm

Vinciarelli and Luettin describe the algorithm in their [2001 paper](http://dx.doi.org/10.1016/S0167-8655(01)00042-3).
Here is a short outline of the algorithm:

![algo](doc/algo.png)
