
from datetime import datetime
import numpy as np
import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()

#얼굴 검출기
face_detector = gaze._face_detector

#학습된 68개 landmarks 모델
predictor = gaze._predictor

#left, right eye landmarks
LEFT_EYE_POINTS = [36, 37, 38,  39, 40, 41]
RIGHT_EYE_POINTS = [42, 43, 44, 45, 46, 47]

webcam = cv2.VideoCapture(0)

while True:
    _, frame = webcam.read()
    cv2.imshow("Demo", frame)

    # 흑백 전환
    image_cvt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image_cvt', image_cvt)

    #filtering
    image_filtered = cv2.bilateralFilter(image_cvt, 10, 15, 15)
    cv2.imshow('image_filtered', image_filtered)

    # erosion
    kernel = np.ones((3, 3), np.uint8)
    image_erosion = cv2.erode(image_filtered, kernel, iterations=3)
    cv2.imshow('image_erosion', image_erosion)

    # 임계처리 : 영상을 흑백으로 분리하여 처리
    image_thresh_binary = cv2.threshold(image_erosion, 50, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('image_thresh_binary', image_thresh_binary)

    #윤곽선 검출
    contours, hierarchy = cv2.findContours(image_thresh_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    """
    얼굴 인식
    """
    image_cvt_face_detector = image_cvt.copy()
    faces = face_detector(image_cvt_face_detector)
    try : cv2.rectangle(image_cvt_face_detector, (faces[0].left(), faces[0].top()), (faces[0].right(), faces[0].bottom()), (0, 0, 255), 2)
    except : pass
    cv2.imshow('face_detector', image_cvt_face_detector)

    """
    얼굴 랜드마크 인식 eye
    """
    height, width = image_cvt.shape[:2]
    black_frame = np.zeros((height, width), np.uint8)
    mask = np.full((height, width), 255, np.uint8)

    try :
        landmarks = predictor(image_cvt_face_detector, faces[0])

        region_left = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in LEFT_EYE_POINTS])
        region_left = region_left.astype(np.int32)

        region_right = np.array([(landmarks.part(point).x, landmarks.part(point).y) for point in RIGHT_EYE_POINTS])
        region_right = region_right.astype(np.int32)

        cv2.fillPoly(mask, [region_left], (0, 0, 0))
        cv2.fillPoly(mask, [region_right], (0, 0, 0))
    except :
        pass

    eye = cv2.bitwise_not(black_frame, mask=mask)
    cv2.imshow('landmarks_eye', eye)


    """
    text
    """
    frame_coords = frame.copy()
    gaze.refresh(frame_coords)
    frame_coords = gaze.annotated_frame()

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    print(datetime.now().strftime('%H:%M:%S.%f'), left_pupil, right_pupil)
    cv2.imshow("frame_coords", frame_coords)


    #키보드 esc 누르면 종료
    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()