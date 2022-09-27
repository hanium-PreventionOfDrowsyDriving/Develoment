# cmd commend
# $ python pi_drowsiness_detection.py --cascade models/haarcascade_frontalface_default.xml --shape-predictor models/shape_predictor_68_face_landmarks.dat --alarm 1


# 필요한 라이브러리 가져오기 
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
    

# A와 B 사이의 유클리드 거리 계산 
def euclidean_dist(ptA, ptB):
	return np.linalg.norm(ptA - ptB)


# 수평과 수직 눈 랜드마크 사이의 거리 비율 계산 함수 
def eye_aspect_ratio(eye):
    # 두 세트의 수직 눈 랜드 마크 (x, y) 좌표 간의 유클리드 거리 계산
	A = euclidean_dist(eye[1], eye[5])
	B = euclidean_dist(eye[2], eye[4])
	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
    # # 수평 눈 랜드 마크 (x, y) 좌표 간의 유클리드 거리 계산
	C = euclidean_dist(eye[0], eye[3])
	# compute the eye aspect ratio
    # 눈 종횡비 계산 
	ear = (A + B) / (2.0 * C)
	# return the eye aspect ratio
    # 눈 종횡비 반환 
	return ear




# 파라메터 구문 분석
ap = argparse.ArgumentParser()

# 얼굴 감지에 사용할 Haar cascade XML 파일 경로
ap.add_argument("-c", "--cascade", required=True,
	help = "path to where the face cascade resides")

# dlib 얼굴 랜드마크 감지 파일 경로
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")

# 졸음이 감지될때 TrafficHat 부저가 사용되는지에 대한 bool형
ap.add_argument("-a", "--alarm", type=int, default=0,
	help="boolean used to indicate if TrafficHat should be used")
# ap.add_argument("-a", "--alarm", type=str, default="", help="path alarm .WAV file")

args = vars(ap.parse_args())



# check to see if we are using GPIO/TrafficHat as an alarm
# 만약 TrafficHat으로 알람을 사용할거라면 
# 제공된 인수가 0보다 크면 부저 알람을 처리하기 위해 TrafficHat함수 가져오기
if args["alarm"] > 0:
	from gpiozero import TrafficHat
	th = TrafficHat()
	print("[INFO] using TrafficHat alarm...")


# 눈의 종횡비가 깜박임을 나타내는 상수와 
# 졸음으로 간주하기 위해 눈의 종횡비 임계값 보다 낮아야 하는 연속 프레임 수에 대한 상수 정의
EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 10

# 프레임 카운터 초기화, 경보음 발생 여부를 나타내는데 사용되는 Bool
COUNTER = 0
ALARM_ON = False



# 얼굴 감지를 위한 Haar cascade 로드
# dlib(HOG)의 내장 감지기를 생성
print("[INFO] 랜드마크 감지기 로딩중..")
detector = cv2.CascadeClassifier(args["cascade"])

# shap_predictor 파일에 대한 경로 
predictor = dlib.shape_predictor(args["shape_predictor"])



# 왼쪽 눈과 오른쪽 눈에 대한 얼굴 랜드 마크의 인덱스 설정
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]



# video stream 초기화 
print("[INFO] 비디오 시작 중..")
vs = VideoStream(src=0).start()

# 파이카메라 용
# vs = VideoStream(usePiCamera=True).start()

# 카메라 센서가 워밍업할 수 있도록 1초 동안 절전 모드
time.sleep(1.0)



# Video Stream 반복
while True:
    # 스레드 비디오 파일 스트림에서 프레임을 가져 와서 크기를 조정한 다음 Grayscale 채널로 변환
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# grayscale fram에서 얼굴 감지
	rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)


# 얼굴 감지 반복
	for (x, y, w, h) in rects:
        # Haar cascade에서 dlib 직사각형 객체를 생성
		rect = dlib.rectangle(int(x), int(y), int(x + w),
			int(y + h))

        # 얼굴 영역의 얼굴 랜드 마크를 결정한 다음 얼굴 랜드 마크 (x, y) 좌표를 NumPy 배열로 변환
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)


		# 왼쪽 및 오른쪽 눈 좌표를 추출하고 좌표를 사용하여 두 눈의 눈 종횡비를 계산
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		
		# 두 눈의 평균 눈 종횡비
		ear = (leftEAR + rightEAR) / 2.0


		# 왼쪽, 오른쪽 눈의 볼록한 부분을 계산후 누을 시각
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)



		# 눈의 종횡비가 깜박임 임계값 미만인지 확인하고, 그렇다면 눈 깜박임 프레임 카운터를 늘림
		if ear < EYE_AR_THRESH:
			COUNTER += 1
			# 눈의 깜박임 수가 연속 깜박임 프레임 임계값 보다 큰 경우 경보음 울림
			
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				# 경보음이 켜져 있지 않으면 켠다
				
				if not ALARM_ON:
					ALARM_ON = True
					# check to see if the TrafficHat buzzer should
					# be sounded
					
					if args["alarm"] > 0:
						th.buzzer.blink(0.1, 0.1, 10,
							background=True)
						
				# 프레임 위에 알람 표시
				
				cv2.putText(frame, "졸음감지!!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				
		# # 그렇지 않으면, 눈 종횡비가 깜박임 임계 값보다 낮지 않으므로 카운터 및 경보음을 재설정
		else:
			COUNTER = 0
			ALARM_ON = False


		# 올바른 눈 종횡비 임계 값 및 프레임 카운터를 디버깅하고 설정하는데 도움이되도록 계산된 눈 종횡비를 프레임에 그린다.
		cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# 프레임 표시
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# q를 누르면 루프 종료
	if key == ord("q"):
		break

# VidioStream 중지 및 윈도우 정리
cv2.destroyAllWindows()
vs.stop()

