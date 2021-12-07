import os
import dlib
import numpy as np
import cv2

# #얼굴 검출기
# face_detector = dlib.get_frontal_face_detector()
#
# #학습된 68개 landmarks 모델
# cwd = os.path.abspath(os.path.dirname(__file__))
# model_path = os.path.abspath(os.path.join(cwd, "trained_models/shape_predictor_68_face_landmarks.dat"))
# predictor = dlib.shape_predictor(model_path)

#left, right eye landmarks
LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

# 이미지 불러오기
test_image = cv2.imread('image.jpg')

#흑백 전환
image_cvt = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

#얼굴 검출기
# faces = face_detector(image_cvt)
# landmarks = predictor(image_cvt, faces[0])

#filtering
image_filtered = cv2.bilateralFilter(image_cvt, 10, 15, 15)

#erosion
kernel = np.ones((3, 3), np.uint8)
image_erosion = cv2.erode(image_filtered , kernel, iterations=3)

#임계처리 : 영상을 흑백으로 분리하여 처리
ret, image_thresh_binary = cv2.threshold(image_erosion, 127, 255, cv2.THRESH_BINARY)
# image_thresh_binary = cv2.bitwise_not(image_thresh_binary)

#윤곽선 검출
# contours, _ = cv2.findContours(image_thresh_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
# contours = sorted(contours, key=cv2.contourArea)
#
# #모멘트 계산
# moments = cv2.moments(contours[-2])
#
# #무게중심
# cx = int(moments['m10'] / moments['m00'])
# cy = int(moments['m01'] / moments['m00'])

red_color = (0, 0, 255)
# image_point = cv2.line(test_image, (cx, cy), (cx, cy), red_color, 5)
# cv2.imshow('image_point', image_point)

cv2.imshow('test_image', test_image)
cv2.imshow('image_cvt', image_cvt)
cv2.imshow('image_filtered', image_filtered)
cv2.imshow('image_erosion', image_erosion)
cv2.imshow('image_thresh_binary', image_thresh_binary)

frame = cv2.imread('image.jpg')
height, width = frame.shape[:2]
black_frame = np.zeros((height, width), np.uint8)
mask = np.full((height, width), 255, np.uint8)

LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

# region = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in points])
# region = region.astype(np.int32)
# self.landmark_points = region
#
# cv2.fillPoly(mask, [region], (0, 0, 0))
# eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)

cv2.waitKey(0)
cv2.destroyAllWindows()