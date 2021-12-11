"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""
from datetime import datetime
import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()

#0 : 웹캠 카페라 지정
#비디오 파일 재생시, 파일 경로와 파일 이름 지정
webcam = cv2.VideoCapture(0)

while True:
    # We get a new frame from the webcam

    """
    ret, frame 반환
    비디오 프레임을 제대로 읽어오면 ret값 True, 실패하면 False
    ret값 체크하여 비디오프레임 제대로 읽었는지 확인
    """
    _, frame = webcam.read()

    gaze.refresh(frame)

    frame = gaze.annotated_frame()
    text = ""

    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Looking right"
    elif gaze.is_left():
        text = "Looking left"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    print(datetime.now().strftime('%H:%M:%S.%f'), left_pupil, right_pupil)
    #19:19:35.348639  left x, y (190, 274) right x, y (267, 278)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
   
webcam.release()
cv2.destroyAllWindows()
