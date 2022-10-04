# python 4.py --haarcascade haarcascade_frontalface_default.xml --shape-predictor shape_predictor_68_face_landmarks.dat

# 필요한 라이브러리 가져오기
from imutils.video import VideoStream  # 비디오 스트림
from imutils import face_utils  # 얼굴 감지 도구 툴
import numpy as np  # 얼굴 랜드마크 위치 및 배열 필요
import argparse  # 감지기 불러오기
import imutils  # 얼굴 감지 관련 라이브러리
import time  # 카메라 센서 워밍
import dlib  # 얼굴 랜드마크
import cv2  # 얼굴 객체 인식
import RPi.GPIO as GPIO
import pyfirmata  # 아두이노
#from Serial2 import Ultra
import Serial2
'''board=pyfirmata.Arduino('/dev/ttyUSB0')
led_builtin=board.get_pin('d:13:o')'''

# A와 B 사이의 유클리드 거리 계산


def euclidean_distance(p, q):
    return np.linalg.norm(p-q)


# 수평과 수직 눈 랜드마크 사이의 거리 비율 계산 함수
def eye_aspect_ratio(eye):
    # 두 세트의 수직 눈 랜드 마크 (x, y) 좌표 간의 유클리드 거리 계산
    P = euclidean_distance(eye[1], eye[5])
    Q = euclidean_distance(eye[2], eye[4])

    # 수평 눈 랜드 마크 (x, y) 좌표 간의 유클리드 거리 계산
    W = euclidean_distance(eye[0], eye[3])

    # 눈 종횡비 계산
    EAR = (P + Q) / (2.0 * W)

    # 눈 종횡비 반환
    return EAR


# 파라미터 구문 분석
parser = argparse.ArgumentParser()

# 얼굴 감지에 사용할 Haar cascade XML 파일 경로
parser.add_argument("-c", "--haarcascade", required=True,
                    help="path to harrcascade xml file")

# dlib 얼굴 랜드마크 감지 파일 경로
parser.add_argument("-p", "--shape-predictor", required=True,
                    help="path to facial landmark predictor")

args = vars(parser.parse_args())

# 눈의 종횡비가 깜박임을 나타내는 상수와
# 졸음으로 간주하기 위해 눈의 종횡비 임계값 보다 낮아야 하는 연속 프레임 수에 대한 상수 정의
EAR_THRESH = 0.23  # 눈 종횡비 임계치 값
EAR_FRAMES = 14  # 프레임수, 값이 높을수록 감지가 늦음.

# 프레임 카운터 초기화, 경보음 발생 여부를 나타내는데 사용되는 Bool
COUNTER = 0
ALARM_ON = False

# 얼굴 감지를 위한 Haar cascade 로드
# dlib의 내장 감지기를 생성
print("얼굴 감지기 로딩중..")
cv_detector = cv2.CascadeClassifier(args["haarcascade"])

# shap_predictor 파일에 대한 경로
dlib_predictor = dlib.shape_predictor(args["shape_predictor"])

# 왼쪽 눈과 오른쪽 눈에 대한 얼굴 랜드 마크의 인덱스 설정
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# video stream 초기화
print("비디오 스트림 시작 중...")
vs = VideoStream(src=0).start()

# 카메라 센서가 워밍업할 수 있도록 1초 동안 절전 모드
time.sleep(1.0)


# ------------------------------------car-------------------------------------------
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    #channel_count = img.shape[2]
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def get_fitline(img, f_lines):  # Finding the representative Line
    try:
        lines = np.squeeze(f_lines)
        print("ch")

        if len(lines.shape) != 1:
            lines = lines.reshape(lines.shape[0] * 2, 2)
            rows, cols = img.shape[:2]
            output = cv2.fitLine(lines, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x, y = output[0], output[1], output[2], output[3]
            # lane change error

            x1, y1 = int(((img.shape[0] - 1) - y) /
                         vy * vx + x), img.shape[0] - 1
            x2, y2 = int(((img.shape[0] / 2 + 70) - y) /
                         vy * vx + x), int(img.shape[0] / 2 + 70)

            result = [x1, y1, x2, y2]

            return result
    except:
        # count up
        return None


# Draw a representative line
def draw_fit_line(img, lines, color=[255, 0, 0], thickness=10):
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)


def drow_the_lines(img, lines):

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=10)


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]


def offset(left, mid, right):

    LANEWIDTH = 3.7
    a = mid - left
    b = right - mid
    width = right - left

    if a >= b:  # driving right off
        offset = a / width * LANEWIDTH - LANEWIDTH / 2.0
    else:  # driving left off
        offset = LANEWIDTH / 2.0 - b / width * LANEWIDTH

    return offset


def process(image):
    # print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    region_of_interest_vertices = [
        (0, height),
        (width/2, height/2),
        (width, height)
    ]
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    canny_image = cv2.Canny(gray_image, 100, 120)
    cropped_image = region_of_interest(canny_image,
                                       np.array([region_of_interest_vertices], np.int32),)

    lines = cv2.HoughLinesP(cropped_image,
                            rho=2,
                            theta=np.pi/180,
                            threshold=50,
                            lines=np.array([]),
                            minLineLength=40,
                            maxLineGap=100)

    line_arr = np.squeeze(lines)

    # Obtaining slope
    slope_degree = (np.arctan2(
        line_arr[:, 1] - line_arr[:, 3], line_arr[:, 0] - line_arr[:, 2]) * 180) / np.pi

    # horizontal slope limit
    line_arr = line_arr[np.abs(slope_degree) < 160]
    slope_degree = slope_degree[np.abs(slope_degree) < 160]
    # vertical slope limit
    line_arr = line_arr[np.abs(slope_degree) > 95]
    slope_degree = slope_degree[np.abs(slope_degree) > 95]
    # Filtered straight line throwout
    L_lines, R_lines = line_arr[(slope_degree > 0),
                                :], line_arr[(slope_degree < 0), :]
    temp = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    L_lines, R_lines = L_lines[:, None], R_lines[:, None]

    # create a representative line
    left_fit_line = get_fitline(temp, L_lines)
    print('left', left_fit_line)
    right_fit_line = get_fitline(temp, R_lines)
    print('right', right_fit_line)
    print(Serial2.Ultra())
    if left_fit_line != None and right_fit_line != None:
        print(right_fit_line[0] - left_fit_line[0])

    color = [255, 0, 0]

    # car detection
    if left_fit_line != None and right_fit_line != None:

        A = [left_fit_line[0], left_fit_line[1]]
        B = [left_fit_line[2], left_fit_line[3]]
        C = [right_fit_line[0], right_fit_line[1]]
        D = [right_fit_line[2], right_fit_line[3]]
        intersection = line_intersection((A, B), (C, D))

        car_mask = np.zeros_like(image)
        match_mask_color = 255
        cv2.fillPoly(car_mask, [np.array(
            [(intersection[0], 50), A, C], np.int32)], match_mask_color)

        car_masked_image = cv2.bitwise_and(image, car_mask)
        car_roi_gray = cv2.cvtColor(car_masked_image, cv2.COLOR_RGB2GRAY)
        cars = car_cascade.detectMultiScale(
            car_roi_gray, 1.4, 1, minSize=(80, 80))

        for (x, y, w, h) in cars:
            print(w, h)
            #pin9.write(0)
            cv2.rectangle(temp, (x, y), (x + w, y + h), (0, 255, 255), 2)

        center = offset(left_fit_line[0], 180, right_fit_line[0])

        print('center', abs(center))
        if abs(center) > 1.5:
            center_x = int(640 / 2.0)
            center_y = int(360 / 2.0)

            thickness = 2

            location = (center_x - 200, center_y - 100)
            font = cv2.FONT_HERSHEY_SIMPLEX  # hand-writing style font
            fontScale = 3.5
            cv2.putText(temp, 'Warning', location, font,
                        fontScale, (0, 0, 255), thickness)
            color = [0, 0, 255]
           # pin9.write(1)

    if left_fit_line != None:
        draw_fit_line(temp, left_fit_line, color)

    if right_fit_line != None:
        draw_fit_line(temp, right_fit_line, color)

    image_with_lines = cv2.addWeighted(temp, 0.8, image, 1, 0.0)

    return image_with_lines


cascade_src = 'cars.xml'
cap = cv2.VideoCapture('change.avi')
car_cascade = cv2.CascadeClassifier(cascade_src)

# -----------------------------------------------------------------------------------------

# Video Stream 반복
while (cap.isOpened()):
    #sum1 = a.examining()
    #count = count + sum1
    # 스레드 비디오 파일 스트림에서 프레임을 가져 와서 크기를 조정한 다음 Grayscale 채널로 변환
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # grayscale fram에서 얼굴 감지
    rects = cv_detector.detectMultiScale(grayscale, scaleFactor=1.1,
                                         minNeighbors=5, minSize=(30, 30),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

# 얼굴 감지 반복
    for (x, y, w, h) in rects:
        # construct a dlib rectangle object from the Haar cascade bounding box
        # Haar cascade에서 dlib 직사각형 객체를 생성
        rect = dlib.rectangle(int(x), int(y), int(x + w),
                              int(y + h))

        # 얼굴 영역의 얼굴 랜드 마크를 결정한 다음 얼굴 랜드 마크 (x, y) 좌표를 NumPy 배열로 변환
        shape = dlib_predictor(grayscale, rect)
        shape = face_utils.shape_to_np(shape)

        # 왼쪽 및 오른쪽 눈 좌표를 추출하고 좌표를 사용하여 두 눈의 눈 종횡비를 계산
        leftEye = shape[left_start:left_end]
        rightEye = shape[right_start:right_end]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        # 두 눈의 평균 눈 종횡비
        EAR = (leftEAR + rightEAR) / 2.0

        # 왼쪽, 오른쪽 눈의 볼록한 부분을 계산후 누을 시각화
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (255, 255, 0), 1)  # B G R
        cv2.drawContours(frame, [rightEyeHull], -1, (255, 255, 0), 1)  # B G R

        # 눈의 종횡비가 깜박임 임계값 미만인지 확인하고, 그렇다면 눈 깜박임 프레임 카운터를 늘림
        if EAR < EAR_THRESH:
            COUNTER += 1

            # 눈의 깜박임 수가 연속 깜박임 프레임 임계값 보다 큰 경우 경보음 울림
            if COUNTER >= EAR_FRAMES:

                # 경보음이 켜져 있지 않으면 켠다
                if not ALARM_ON:
                    ALARM_ON = True

                    # led_builtin.write(1)

                    # check to see if the TrafficHat buzzer should
                    # be sounded
                    '''if args["alarm"] > 0:
						th.buzzer.blink(0.1, 0.1, 10,
							background=True)'''

                # 프레임 위에 알람 표시
                cv2.putText(frame, "WAKE UP!!", (300, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Frame_width(10, 30)

        # 그렇지 않으면, 눈 종횡비가 깜박임 임계 값보다 낮지 않으므로 카운터 및 경보음을 재설정
        else:
            COUNTER = 0
            ALARM_ON = False

            # led_builtin.write(0)

        # 올바른 눈 종횡비 임계 값 및 프레임 카운터를 디버깅하고 설정하는데 도움이되도록 계산된 눈 종횡비를 프레임에 그린다.
        cv2.putText(frame, "EAR: {:.3f}".format(EAR), (300, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)  # Fram_weight(300, 30)

    # 프레임 표시
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    '''# q를 누르면 루프 종료
    if key == ord("q"):
        break
    '''
# -------------------------------car--------------------------------
    ret, frame = cap.read()

    if (type(frame) == type(None)):
        break

    frame = process(frame)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# VidioStream 중지 및 윈도우 정리
cv2.destroyAllWindows()
vs.stream.release()  # 활성된 카메라 끄기 - 실시간 찰영시 주석처리해도 됨
vs.stop()

