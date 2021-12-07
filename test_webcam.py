import numpy as np
import cv2

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

    #키보드 esc 누르면 종료
    if cv2.waitKey(1) == 27:
        break

print(type(contours))
print(type(hierarchy))

webcam.release()
cv2.destroyAllWindows()