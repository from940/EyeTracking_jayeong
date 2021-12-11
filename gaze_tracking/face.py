import math
import numpy as np
import cv2
from .pupil import Pupil

class Face(object) :
    """
    얼굴 좌표 인식 후 반환
    """

    FACE_TOP = [21, 22]
    FACE_BOTTOM = [7,8,9]
    FACE_LETF = [0, 1, 2, 3]
    FACE_RIGHT = [13, 14, 15, 16]

    def __init__(self, original_frame, landmarks, calibration):
