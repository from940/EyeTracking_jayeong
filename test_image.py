
import numpy as np
import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()

#얼굴 검출기
face_detector = gaze._face_detector

#학습된 68개 landmarks 모델
predictor = gaze._predictor

#left, right eye landmarks
LEFT_EYE_POINTS = [36, 37, 38, 39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

# 이미지 불러오기
test_image = cv2.imread('image.jpg')
# cv2.imshow('test_image', test_image)

# #흑백 전환
image_cvt = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('image_cvt', image_cvt)

#얼굴 검출기
faces = face_detector(image_cvt)
#사진에 있는 얼굴을 인식함
#3명이 있는 사진이라면 len(faces) = 3
#1명이 있는 사진이라면 len(faces) = 1
#faces[0].left(), faces[0].top(), faces[0].right(), faces[0].bottom()


# win = dlib.image_window()
# win.set_image(image_cvt)
# win.add_overlay(faces)
# cv2.imshow('image_cvt', image_cvt)
# cv2.imwrite("output.jpg", image_cvt)

# print(len(faces))
landmarks = predictor(image_cvt, faces[0])

#filtering
image_filtered = cv2.bilateralFilter(image_cvt, 10, 15, 15)
# cv2.imshow('image_filtered', image_filtered)

#erosion
kernel = np.ones((3, 3), np.uint8)
image_erosion = cv2.erode(image_filtered , kernel, iterations=3)
# cv2.imshow('image_erosion', image_erosion)


#임계처리 : 영상을 흑백으로 분리하여 처리
ret, image_thresh_binary = cv2.threshold(image_erosion, 127, 255, cv2.THRESH_BINARY)
# image_thresh_binary = cv2.bitwise_not(image_thresh_binary)
# cv2.imshow('image_thresh_binary', image_thresh_binary)

#
# #윤곽선 검출
# # contours, _ = cv2.findContours(image_thresh_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
# # contours = sorted(contours, key=cv2.contourArea)
# #
# # #모멘트 계산
# # moments = cv2.moments(contours[-2])
# #
# # #무게중심
# # cx = int(moments['m10'] / moments['m00'])
# # cy = int(moments['m01'] / moments['m00'])
#
# red_color = (0, 0, 255)
# # image_point = cv2.line(test_image, (cx, cy), (cx, cy), red_color, 5)
# # cv2.imshow('image_point', image_point)
#
cv2.imshow('test_image', test_image)
cv2.imshow('image_cvt', image_cvt)
cv2.imshow('image_filtered', image_filtered)
cv2.imshow('image_erosion', image_erosion)
cv2.imshow('image_thresh_binary', image_thresh_binary)


#
region_left = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in LEFT_EYE_POINTS])
region_right = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in RIGHT_EYE_POINTS])
region_left = region_left.astype(np.int32)
region_right = region_right.astype(np.int32)
landmark_points_left = region_left
landmark_points_right = region_right

frame = cv2.imread('image.jpg')

height, width = frame.shape[:2]
black_frame = np.zeros((height, width), np.uint8)
mask = np.full((height, width), 255, np.uint8)

cv2.fillPoly(mask, [region_left], (0, 0, 0))
cv2.fillPoly(mask, [region_right], (0, 0, 0))

# eye = cv2.bitwise_not(black_frame, frame.copy(), mask=mask)
eye = cv2.bitwise_not(black_frame,  mask=mask)
cv2.imshow('bitwise_not', eye)

margin = 5
left_min_x = np.min(region_left[:, 0]) - margin
left_max_x = np.max(region_left[:, 0]) + margin
left_min_y = np.min(region_left[:, 1]) - margin
left_max_y = np.max(region_left[:, 1]) + margin

frame = eye[left_min_y:left_max_y, left_min_x:left_max_x]
cv2.imshow('what is margin?', frame)

origin = (left_min_x, left_min_y)
print('origin', origin)

height, width = frame.shape[:2]
print(height, width)
center = (width / 2, height / 2)
print('center',center)

# print(type(region_left))
# print(len(region_left))
# print(LEFT_EYE_POINTS)
# print(region_left)
# print(region_left[:, 0])
# print( region_left[:, 1])

# print(region_right)
# print(region_left[:, 0], region_left[:, 1])

cv2.waitKey(0)
cv2.destroyAllWindows()



