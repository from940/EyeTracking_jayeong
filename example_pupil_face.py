
import cv2
from datetime import datetime
from face_tracking import FaceTracking
from gaze_tracking import GazeTracking

gaze = GazeTracking()
face = FaceTracking()

# 0 : 웹캠 카페라 지정
# 비디오 파일 재생시, 파일 경로와 파일 이름 지정
webcam = cv2.VideoCapture(0)

while True:
    # We get a new frame from the webcam

    """
    ret, frame 반환
    비디오 프레임을 제대로 읽어오면 ret값 True, 실패하면 False
    ret값 체크하여 비디오프레임 제대로 읽었는지 확인
    """
    _, frame = webcam.read()

    frame_for_gaze = frame.copy()
    gaze.refresh(frame_for_gaze)
    frame_for_gaze = gaze.annotated_frame()
    cv2.imshow("frame for gaze", frame_for_gaze)

    frame_for_face = frame.copy()
    face.refresh(frame_for_face)
    frame_for_face = face.annotated_frame()

    cv2.imshow("frame for face", frame_for_face)

    # 좌표 출력

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    left_coords = face.face_left_coords()
    right_coords = face.face_right_coords()
    top_coords = face.face_top_coords()
    bottom_coords = face.face_bottom_coords()

    print(datetime.now().strftime('%H:%M:%S.%f'), left_coords, right_coords, top_coords, bottom_coords , left_pupil, right_pupil)

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()