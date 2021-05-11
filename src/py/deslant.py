from collections import namedtuple
from typing import Tuple

import cv2
import numpy as np

DeslantRes = namedtuple('DeslantRes', 'img, candidates')
Candidate = namedtuple('Candidate', 'shear_val, score')


def get_shear_vals(min_val: float, max_val: float, step: float) -> Tuple[float]:
    """Compute shear values in given range."""
    return tuple(np.arange(min_val, max_val + step, step))


def shear_img(img: np.ndarray, s: float, bg_color: int, interpolation=cv2.INTER_NEAREST) -> np.ndarray:
    """
    Shears the given image.
    """
    h, w = img.shape
    offset = h * s
    w = w + int(abs(offset))
    tx = max(-offset, 0)

    shear_transform = np.asarray([[1, s, tx], [0, 1, 0]], dtype=float)
    img_sheared = cv2.warpAffine(img, shear_transform, (w, h), flags=interpolation, borderValue=bg_color)

    return img_sheared


def deslant(img: np.ndarray,
            shear_vals: Tuple[float] = get_shear_vals(-2, 2, 0.25),
            bg_color=255) -> DeslantRes:
    """
    Deslants the image by applying a shear transform.

    The function searches for a shear transform that yields many long connected vertical lines.

    Args:
        img: The image to be deslanted with text in black and background in white.
        shear_vals: The shear values to be evaluated.
        bg_color: Color that is used to fill the gaps of the sheared image that is returned.

    Returns:
        Object of DeslantRes, holding the deslanted image and the candidates with shear value and score.
    """
    assert img.ndim == 2
    assert img.dtype == np.uint8

    # apply Otsu's threshold method to inverted input image
    img_mask = cv2.threshold(255 - img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] // 255

    # compute score for all shear values
    candidates = []
    for s in shear_vals:
        img_sheared = shear_img(img_mask, s, 0)
        h = img_sheared.shape[0]

        img_sheared_mask = img_sheared > 0
        first_fg_px = np.argmax(img_sheared_mask, axis=0)
        last_fg_px = h - np.argmax(img_sheared_mask[::-1], axis=0)
        num_fg_px = np.sum(img_sheared_mask, axis=0)

        dist_fg_px = last_fg_px - first_fg_px
        col_mask = np.bitwise_and(num_fg_px > 0, dist_fg_px == num_fg_px)
        masked_dist_fg_px = dist_fg_px[col_mask]

        score = np.sum(masked_dist_fg_px ** 2)
        candidates.append(Candidate(s, score))

    # select shear value that yields highest score
    res_shear_val = sorted(candidates, key=lambda c: c.score, reverse=True)[0].shear_val
    res_img = shear_img(img, res_shear_val, bg_color, cv2.INTER_LINEAR)
    return DeslantRes(res_img, candidates)
