

import numpy as np

class Part(object):
    """
    얼굴 일부분 인식 후 x,y 좌표 반환
    """

    FACE_TOP = [21, 22]
    FACE_BOTTOM = [7, 8, 9]
    FACE_LEFT = [0, 1, 2, 3]
    FACE_RIGHT = [13, 14, 15, 16]

    def __init__(self, landmarks, side):
        self.frame = None
        self.region_list = None
        self.x = None
        self.y = None

        # original_frame으로 cvt 웹캠 영상 받음
        self._analyze(landmarks, side)


    def region(self, landmarks, points):
        """landmarks 영역 반환"""
        region_list = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
        region_list = region_list.astype(np.int32)
        return region_list

    def location(self, region_list):
        """좌표 반환"""
        x = int(sum([point[0] for point in region_list]) / len(region_list))
        y = int(sum([point[1] for point in region_list]) / len(region_list))
        return x, y

    def _analyze(self, landmarks, side):
        if side == 0:
            points = self.FACE_LEFT
        elif side == 1:
            points = self.FACE_RIGHT
        elif side == 2:
            points = self.FACE_TOP
        elif side == 3:
            points = self.FACE_BOTTOM
        else:
            return

        self.region_list = self.region(landmarks, points)

        self.x, self.y = self.location(self.region_list)