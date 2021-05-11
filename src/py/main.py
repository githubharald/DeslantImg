import argparse
from typing import List

import cv2
import matplotlib.pyplot as plt
from path import Path

from deslant import deslant


def get_img_files(data_dir: Path) -> List[Path]:
    """Returns all image files contained in a folder"""
    res = []
    for ext in ['*.png', '*.jpg', '*.bmp']:
        res += Path(data_dir).files(ext)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('../../data/'))
    parser.add_argument('--bg_color', type=int, default=255)
    parsed = parser.parse_args()

    for fn_img in get_img_files(parsed.data):
        img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
        res = deslant(img, bg_color=parsed.bg_color)

        plt.subplot(221)
        plt.imshow(img, cmap='gray')
        plt.title('Original')

        plt.subplot(222)
        plt.imshow(res.img, cmap='gray')
        plt.title('Deslanted')

        plt.subplot(212)
        shear_vals = [c.shear_val for c in res.candidates]
        score_vals = [c.score for c in res.candidates]
        plt.stem(shear_vals, score_vals)
        plt.title('Score values')

        plt.show()


if __name__ == '__main__':
    main()
