
from __future__ import division
import os
import cv2
import dlib
from pupil import Pupil
from eye import Eye
from calibration import Calibration
from gaze_tracking import GazeTracking

face_detector = dlib.get_frontal_face_detector()

cwd = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.abspath(os.path.join(cwd, "gaze_tracking/trained_models/shape_predictor_68_face_landmarks.dat"))
predictor = dlib.shape_predictor(model_path)

# 이미지 불러오기
test_image = cv2.imread('image.jpg')
frame = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
faces = face_detector(frame)
landmarks = predictor(frame, faces[0])

calibration = Calibration()
threshold = calibration.threshold(0)
pupil = Pupil(frame, threshold)
LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
points = LEFT_EYE_POINTS


contours, _ = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
contours = sorted(contours, key=cv2.contourArea)
moments = cv2.moments(contours[-2])
left_x = int(moments['m10'] / moments['m00'])
left_y = int(moments['m01'] / moments['m00'])

cv2.drawContours(frame, contours, -1 , (0,255,0), thickness=None, lineType=5)
# image_point = cv2.line(test_image, (cx, cy), (cx, cy), red_color, 5)
cv2.imshow('image_point', frame)

# contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
# contours = sorted(contours, key=cv2.contourArea)
# moments = cv2.moments(contours[-2])
# right_x = int(moments['m10'] / moments['m00'])
# right_y = int(moments['m01'] / moments['m00'])