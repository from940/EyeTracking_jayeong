
import numpy as np
import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()

# 얼굴 검출기
face_detector = gaze._face_detector

# 학습된 68개 landmarks 모델
predictor = gaze._predictor

# 이미지 불러오기
test_image = cv2.imread('image.jpg')

# # 흑백 전환
# image_cvt = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

"""
FaceDetector
사진에 있는 얼굴을 인식함
3명이 있는 사진이라면 len(faces) = 3
1명이 있는 사진이라면 len(faces) = 1
faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom()
"""
faces = face_detector(test_image)
cv2.rectangle(test_image, (faces[0].left(), faces[0].top()), (faces[0].right(),faces[0].bottom()), (0,0,255), 2)
cv2.imshow('face_detector', test_image)
print('left, top, right, bottom: ', faces[0].left(), faces[0].top(), faces[0].right(),faces[0].bottom())


"""
landmarks
"""
landmarks = predictor(test_image, faces[0])
print(type(landmarks))


cv2.waitKey(0)
cv2.destroyAllWindows()