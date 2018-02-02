# Deslanting Algorithm
This algorithm sets the (handwritten) text in images upright, i.e. it removes the cursive writing style.
It can be used as a preprocessing step in handwritten text recognition.
The following image (taken from IAM dataset \[2]) shows input (left) and output (right) of the algorithm for the test file (```data/test.png```).
![deslanting](./doc/deslanting.png)


## Getting started
Build on Linux:
```g++ --std=c++11 main.cpp DeslantImg.cpp -o DeslantImg `pkg-config --cflags --libs opencv` ```

And run:
```./DeslantImg```

This should read the image ```data/test.png``` and output the deslanted image ```out.png```.
Implemented in C++ using OpenCV3.
Tested on Windows and Linux using the IAM, Bentham and Ratsprotokolle datasets.


## Algorithm 
Vinciarelli and Luettin describe the algorithm in their paper \[1].
Here is a short outline of the algorithm:
![algo](./doc/algo.png)


## Documentation
Call function ```deslantImg(img, bgcolor)``` with the input image (grayscale) and the background color (to fill empty image space).
It returns the deslanted image.

The following code reads an image, deslants it and finally saves it (see ```main.cpp```):
```
// read grayscale image
const cv::Mat img = cv::imread("data/test.png", cv::IMREAD_GRAYSCALE);

// deslant it
cv::Mat res = deslantImg(img, 255);

// and save the result
cv::imwrite("out.png", res);
```


## References

\[1] Vinciarelli and Luettin - A new normalization technique for cursive handwritten words

\[2] http://www.fki.inf.unibe.ch/databases/iam-handwriting-database
