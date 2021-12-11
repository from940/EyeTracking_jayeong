import numpy as np
import cv2


class Pupil(object):
    """
    This class detects the iris of an eye and estimates
    the position of the pupil
    """

    def __init__(self, eye_frame, threshold):
        self.iris_frame = None

        """
        임계값
        사용자가 특정 수치값을 정해놓으면 그 기준값을 통해 값을 도출한다.
        openCV에서 Thresholding을 사용할 땐 grayscale 흑백으로 작업한다.
        """
        self.threshold = threshold

        self.x = None
        self.y = None

        #iris의 이미지 무게중심 x좌표, y좌표
        self.detect_iris(eye_frame)

    @staticmethod
    def image_processing(eye_frame, threshold):
        """Performs operations on the eye frame to isolate the iris

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
            threshold (int): Threshold value used to binarize the eye frame

        Returns:
            A frame with a single element representing the iris
        """

        """
        필터 적용하기
        OpenCV의 필터. 선형으로 처리되지 않고 엣지와 노이즈를 줄여 부드러운 영상을 만든다.
        bilateralFilter(src, dst, d, sigmaColor, sigmaSpace)
        src : 입력 이미지
        dst : 출력 이미지
        d : 필터링에 이용하는 이웃한 픽셀의 지름을 정의 불가능한경우 sigmaspace 를 사용
        sigmaColor : 컬러공간의 시그마공간 정의, 클수록 이웃한 픽셀과 기준색상의 영향이 커진다
        sigmaSpace : 시그마 필터를 조정한다. 값이 클수록 긴밀하게 주변 픽셀에 영향을 미친다. d>0 이면 영향을 받지 않고, 그 외에는 d 값에 비례한다.
        """
        new_frame = cv2.bilateralFilter(eye_frame, 10, 15, 15)

        """공간 영역 필터링
                convolution 연산(합성곱 연산)을 통해 이미지에서 특징을 추출한다.
                연산 대상 픽셀과 그 주변 픽셀을 활용해 새로운 픽셀값을 얻는다.
                주변 픽셀을 어느 범위까지 활용할지, 연산은 어떻게 할지를 결정하는 역할이 커널이다. 
                커널을 윈도우 window, 필터 filter, 마스크 mask 라고도 부른다.
                3x3 크기 배열인 컨볼루션 커널. 
                np.unit8 : 양수만 표현 가능한 정수형 자료형
                """
        kernel = np.ones((3, 3), np.uint8)

        """
        모폴로지 연산 - 침식
        이미지를 침식시키는 erosion과 팽창시키는 dilation
        이미지 erosion의 원리는 전경 foreground의 이미지의 경계부분을 배경 background 이미지로 전환시켜
        전경이 되는 이미지를 가늘게 한다. 
        커널이 이미지를 읽고 커널에 놓인 원본 이미지의 픽셀값이 모두 1일 경우 1을 반환, 그렇지 않으면 0을 반환
        cv2.erode(img, kernel, iteratoins=1)
        img : erosion 수행할 원본 이미지
        kernel : erosion 위한 커널
        iterations : erosion 반복 횟수
        """
        new_frame = cv2.erode(new_frame, kernel, iterations=3)

        """
        이미지 임계처리
        cv2.threshold(img, threshold_value, value, flag)
        img : 대상 grayscale 이미지
        threshold_value : 사용자 지정 픽셀 임계값
        value : 지정한 임계값 기준으로 크고 작을 때 적용할 적용값
        flag : 임계값 적용 스타일
        cv2.THRESH_BINARY :픽셀값이 지정한 임계값보다 크면 value 값을 부여, 작으면 0을 부여 
        """
        new_frame = cv2.threshold(new_frame, threshold, 255, cv2.THRESH_BINARY)[1]

        return new_frame

    def detect_iris(self, eye_frame):
        """Detects the iris and estimates the position of the iris by
        calculating the centroid.

        Arguments:
            eye_frame (numpy.ndarray): Frame containing an eye and nothing else
        """
        self.iris_frame = self.image_processing(eye_frame, self.threshold)

        """
        윤곽선 검출
        contours, hierarchy = cv2.findContours(이진화 이미지, 검색 방법, 근사화 방법)
        countours 윤곽선 : numpy 구조 배열로 검출된 윤곽선의 지점들
        hierarchy 계층구조 : 윤곽선의 계층 구조. 각 윤곽선에 해당하는 속성 정보들
        cv2.RETR_TREE : 모든 윤곽선 검출하고 계층 구조를 모두 형성한다.(Tree 구조)
        cv2.CHAIN_APPROX_NONE : 윤곽점들의 모든 점을 반환한다. 
        """
        contours, _ = cv2.findContours(self.iris_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[-2:]
        #contourArea는 폐곡선(닫힌 곡선) 형태의 contour로 둘러싸인 부분의 면적
        contours = sorted(contours, key=cv2.contourArea)

        """
        모멘트 : 영상의 형태를 표현하는 일련의 실수 값
        윤곽선이나 이미지의 0차 모멘트부터 3차 모멘트까지 계산
        cv2.moments(배열, 이진화 이미지)를 활용해 윤곽선에서 모멘트를 계산한다.
        
        이미지 모멘트는 영상 픽셀의 강도에 대한 특정한 가중평균, 
        일반적으로 어떤 물체의 고유한 특성이나 해석을 할 수 있는 기능을 말한다.
        무게 중심(mass center)
        
        오브젝트의 중심을 찾는데 사용된다. 
        0차 모멘트 : m00
        1차 모멘트 : m10, m01 
        """
        try:
            moments = cv2.moments(contours[-2])
            #이미지의 무게 중심 x좌표, y좌표
            self.x = int(moments['m10'] / moments['m00'])
            self.y = int(moments['m01'] / moments['m00'])
        except (IndexError, ZeroDivisionError):
            pass


