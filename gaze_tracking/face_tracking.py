
from __future__ import division
import os
import cv2
import dlib
from part_of_face import Part


class FaceTracking(object):
    """
    얼굴 좌표 인식 후 반환
    """

    def __init__(self):
        self.frame = None

        self.face_left = None
        self.face_right = None
        self.face_top = None
        self.face_bottom = None

        self._face_detector = dlib.get_frontal_face_detector()

        cwd = os.path.abspath(os.path.dirname(__file__))
        model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
        self._predictor = dlib.shape_predictor(model_path)

    @property
    def face_located(self):
        """Check that the face have been located"""
        try:
            int(self.face_left.x)
            int(self.face_left.y)
            int(self.face_right.x)
            int(self.face_right.y)

            int(self.face_top.x)
            int(self.face_top.y)
            int(self.face_bottom.x)
            int(self.face_bottom.y)
            return True
        except Exception:
            return False

    def refresh(self, frame):
        """Refreshes the frame and analyzes it.

        Arguments:
            frame (numpy.ndarray): 원본 웹캠 영상
        """
        self.frame = frame
        self._analyze()

    # 좌표
    def face_left_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.face_located:
            x = self.face_left.x
            y = self.face_left.y
            return (x, y)

    # 좌표
    def face_right_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.face_located:
            x = self.face_right.x
            y = self.face_right.y
            return (x, y)

    def face_top_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.face_located:
            x = self.face_top.x
            y = self.face_top.y
            return (x, y)

    def face_bottom_coords(self):
        """Returns the coordinates of the left pupil"""
        if self.face_located:
            x = self.face_bottom.x
            y = self.face_bottom.y
            return (x, y)

    def _analyze(self):
        """Detects the face and initialize Eye objects"""
        # 원본 웹캠 frame을 흑백으로 변환
        frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
        # dlib의 정면 얼굴 검출기
        faces = self._face_detector(frame)

        try:
            landmarks = self._predictor(frame, faces[0])

            self.face_left = Part(landmarks, 0)
            self.face_right = Part(landmarks, 1)
            self.face_top = Part(landmarks, 2)
            self.face_bottom = Part(landmarks, 3)

        except IndexError:
            self.face_left = None
            self.face_right = None
            self.face_top = None
            self.face_bottom = None


    def annotated_frame(self):
        """
        얼굴 point 4개 프레임 반환
        """
        frame = self.frame.copy()

        if self.face_located:
            color = (0, 255, 0)
            face_left_x, face_left_y = self.face_left.x , self.face_left.y
            face_right_x, face_right_y = self.face_right.x , self.face_right.y
            face_top_x, face_top_y = self.face_top.x, self.face_top.y
            face_bottom_x, face_bottom_y = self.face_bottom.x, self.face_bottom.y

            cv2.circle(frame, (face_left_x, face_left_y) , 2,  color)
            cv2.circle(frame, (face_right_x, face_right_y),2,  color)
            cv2.circle(frame, (face_top_x, face_top_y), 2, color)
            cv2.circle(frame, (face_bottom_x, face_bottom_y), 2,  color)

        return frame


