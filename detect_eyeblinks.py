# TestFile : blink_detection_Number.mp4 -> Number 자리에 숫자 대입(e.g. _1) 
# $ python detect_eyeblinks.py --shape-predictor models/shape_predictor_68_face_landmarks.dat --video video/blink_detection_1.mp4
# # 만약 웹캠을 이용한다면, line 65, 69 uncomment
# $ python detect_eyeblinks.py --shape-predictor models/shape_predictor_68_face_landmarks.dat

from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


def eye_aspect_ratio(eye):
	# 두 수직 눈 랜드 마크 (x, y) 좌표 간의 유클리드 거리 계산
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	
	# 수평 눈 랜드 마크 (x, y) 좌표 간의 유클리드 거리 계산
	C = dist.euclidean(eye[0], eye[3])
	
	# 눈 종횡비 계산
	ear = (A + B) / (2.0 * C)
	
	# 눈 종횡비 반환
	return ear

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())

# 눈의 종횡비(깜박임) 값을 나타내는 변수
# 임계값 보다 낮아야 하는 연속 프레임 수에 대한 변수
EYE_AR_THRESH = 0.23
# 눈을 빠르게 감아도 카운트하고 싶으면 값 하강
# 아니라면 값 증가
EYE_AR_CONSEC_FRAMES = 1.8 # 3

# 초기 프레임 카운트
# 눈 깜박임 총 횟수
COUNTER = 0
TOTAL = 0

# dlib의 얼굴 탐지기 (HOG 기반) 로드 및 얼굴 랜드 마크 변수 생성
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# 왼쪽 눈과 오른쪽 눈에 대한 얼굴 랜드 마크의 인덱스 설정
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# 비디오 스트림 초기화 및 시작
print("[INFO] starting video stream thread...")
# # 동영상 부분
vs = FileVideoStream(args["video"]).start()
fileStream = True
# 웹캠 부분
# vs = VideoStream(src=0).start()
# # 파이카메라 부분
# vs = VideoStream(usePiCamera=True).start()
# # 60, 62 코드를 사용한다면 주석 해제 - 비디오 파일은 읽지 않기 위해
# fileStream = False
time.sleep(1.0)


# 비디오 스트림 반복
while True:
	# 파일 비디오 스트림인 경우 처리 할 버퍼에 프레임이 더 남아 있는지 확인
	if fileStream and not vs.more():
		break
	# 스레드 비디오 파일 스트림에서 프레임을 가져 와서 크기를 조정한 다음 Grayscale 채널로 변환
	frame = vs.read()
	frame = imutils.resize(frame, width=450)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# Grauscale 프레임에서 얼굴 감지
	rects = detector(gray, 0)
	
	
	# 얼굴 감지 반복
	for rect in rects:
		# 얼굴영역을 찾기위해 얼굴 랜드마크 결정
		# 얼굴 랜드 마크 (x, y) 좌표를 NumPy 배열로 변환
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		
		# 왼쪽 및 오른쪽 눈 좌표를 추출한 다음 좌표를 사용하여 두 눈의 눈 종횡비 계산
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		
		# 두 눈의 평균 눈 종횡비
		ear = (leftEAR + rightEAR) / 2.0
		
		
		# 랜드마크 중 눈 부분만 추출
		# 왼쪽 눈과 오른쪽 눈의 눈꺼플을 계산 한 다음 각 눈을 시각화
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
		
		# 눈의 종횡비가 깜박임 임계값 보다 낮은지 확인하고, 그렇다면 눈 깜박임 프레임 카운터 증가
		if ear < EYE_AR_THRESH:
			COUNTER += 1
			
		# 그렇지 않으면, 눈의 종횡비가 깜박임 임계값 보다 낮지 않음
		else:
			# 눈의 깜박임 수가 깜박임 프레임 임계값 보다 큰 경우 총 깜박임 횟수 증가
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
			# 눈 깜박임 프레임 카운터 초기화
			COUNTER = 0


# 프레임 위에 계산된 총 눈 깜박임 횟수, 눈 종횡비 표시
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
	# 프레임 보여줌
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# q로 반복 종료
	if key == ord("q"):
		break
	
# cleanup
cv2.destroyAllWindows()
vs.stream.release() # 활성된 카메라 끄기 - 실시간 찰영시 주석처리해도 됨
vs.stop()