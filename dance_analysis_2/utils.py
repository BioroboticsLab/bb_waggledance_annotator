# utils.py

import os
import csv
import cv2 as cv
import numpy as np
from typing import Tuple


def get_output_filename(filepath: str) -> str:
    """
    Generates the output CSV filename based on the input video file path.

    Args:
        filepath (str): The path to the input video file.

    Returns:
        str: The path to the output CSV file for annotations.
    """
    # Extract the base file name without the extension
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    # Get the directory of the input file
    directory = os.path.dirname(filepath)
    # Create the output filename in the same directory
    output_filename = os.path.join(directory, f"{base_name}_waggle_annotations.csv")
    return output_filename


def get_csv_writer_options() -> dict:
    """
    Provides standard options for CSV writing.

    Returns:
        dict: A dictionary of CSV writer options.
    """
    return dict(quoting=csv.QUOTE_ALL, quotechar='"', delimiter=",")


def calculate_time(current_frame: int, video_fps: float) -> float:
    """
    Calculates the timestamp in seconds for a given frame number.

    Args:
        current_frame (int): The current frame index.
        video_fps (float): The frames per second of the video.

    Returns:
        float: The timestamp in seconds, rounded to two decimal places.
    """
    frame_timestamp_sec = current_frame / video_fps
    return round(frame_timestamp_sec, 2)


def fit_image_to_aspect_ratio(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """
    Fits the aspect ratio of an image to fit within given width and height constraints.
    Adds black borders as necessary to retain the original pixel coordinates.

    Args:
        image (np.ndarray): The input image.
        target_width (int): The target width for the image.
        target_height (int): The target height for the image.

    Returns:
        np.ndarray: The image adjusted to fit the target aspect ratio.
    """
    image_height, image_width = image.shape[:2]

    target_aspect = target_width / target_height
    image_aspect = image_width / image_height

    bottom_border = 0
    right_border = 0

    if target_aspect == image_aspect:
        pass
    elif target_aspect > image_aspect:
        right_border = int(image_height * target_aspect) - image_width
    else:
        bottom_border = int(image_width / target_aspect) - image_height

    if bottom_border > 0 or right_border > 0:
        image = cv.copyMakeBorder(
            image, 0, bottom_border, 0, right_border, cv.BORDER_CONSTANT, value=[0, 0, 0]
        )

    return image


def calc_frames_to_move(k32: int, video_fps: float, debug: bool = False) -> int:
    """
    Determines the number of frames to move based on key input and modifiers.

    Args:
        k32 (int): The key code from OpenCV's waitKeyEx function.
        video_fps (float): The frames per second of the video.
        debug (bool): If True, prints debug information.

    Returns:
        int: The number of frames to move forward or backward.
    """
    # Modifiers are actually 17, 18, 19 bits away
    ctrl = (k32 & (1 << 18)) >> 18
    shift = (k32 & (1 << 16)) >> 16
    alt = (k32 & (1 << 19)) >> 19
    modifiers = f"{ctrl:0x} {shift:0x} {alt:0x}"
    if k32 != -1 and debug:
        key = k32 & 0xff
        print(
            "\033[34m"
            f"KEY: {k32} 0x{k32:2x} |{key} 0x{key:2x} |{modifiers}|"
            "\033[0m"
        )
    nframes = 1
    if shift:
        nframes = int(video_fps * 1)
    elif ctrl:
        nframes = int(video_fps * 5)
    elif alt:
        nframes = int(5)

    return nframes