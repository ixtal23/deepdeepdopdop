import logging as log

from pathlib import Path
from cv2 import imread, imwrite

from .types import Frame

def read_image(file_path : Path) -> Frame:
    log.info(f'Read image from file {file_path}')

    frame = imread(str(file_path))
    if frame.any():
        height, width = frame.shape[:2]
        log.info(f'Image was read: width={width}, height={height}')
    else:
        log.error('Failed to read image file')

    return frame

def write_image(file_path : Path, frame : Frame) -> None:
    log.info(f'Write image to file {file_path}')

    imwrite(str(file_path), frame)
