# python video_facial_landmarks.py --shape-predictor models/shape_predictor_68_face_landmarks.dat

from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2

# 입력받을 인자 값 설정
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# dlib landmark 감지기를 불러온다.
# the facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# 비디오 스트림을 초기화하고 카메라 센서를 준비한다.
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# 비디오 스트림에서 프레임을 반복
while True:
	# 비디오 설정
	# 400 사이즈로 설정 및 회색으로 변환
	frame = vs.read()
	frame = imutils.resize(frame, width=400)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	# grayscale framed - 얼굴감지를 수행한다.
	rects = detector(gray, 0)
	
	
	# 얼굴 감지기 반복 수행
	for rect in rects:
		
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
		
		for (x, y) in shape:
			cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
	  
	# 프레임 보여줌
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
 
	# q로 반복 종료
	if key == ord("q"):
		break
	
	

cv2.destroyAllWindows()
vs.stream.release() # 활성된 카메라 끄기 - 실시간 찰영시 주석처리해도 됨
vs.stop()