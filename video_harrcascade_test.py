# python video_harrcascade_test.py --cascade models/haarcascade_frontalface_default.xml

from imutils.video import VideoStream # 웹캠에 접근 허용 
import argparse
import imutils
import time
import cv2


ap = argparse.ArgumentParser() # 프로그램을 실행시에 명령어줄에 인수를 받아 처리를 간단히 할 수 있도록 하는 표준 라이브러리
# 입력받을 인자 값 설정 
ap.add_argument("-c", "--cascade", type=str, # --cascade : 디스크(디렉토리)에 있는 훈련된 Haarcascade를 가리킴
	default="haarcascade_frontalface_default.xml",
	help="path to haar cascade face detector") # 해당 인수의 설명
args = vars(ap.parse_args()) # computer enviroment 오류 잡기위해'' 추가


# haar cascade 얼굴 감지기를 불러온다.
print("[INFO] loading face detector...") # 안내문
detector = cv2.CascadeClassifier(args["cascade"]) # cascade가 가리키는 'haarcascade'를 Classifier(분류기)로 사용



print("[INFO] starting video stream...")
vs = VideoStream(src=0).start() # 비디오 스트림 시작 
time.sleep(2.0) # 2초간 프로세스 일시정지 


# 비디오 스트림에서 프레임을 반복
while True: # 초당 지정된 프레임 수만큼 비디오 화면 생성 
	frame = vs.read() # 카메라로부터 비디오 스트림 읽기 
	frame = imutils.resize(frame, width=500) # 사이즈를 500으로 조정, imutils: 이미지 파일 및 비디오 스트림 파일을 처리하는 유틸리티용 라이브러리 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 프레임을 회색깔로 변경 

	# 얼굴감지를 수행한다.
	 # scaleFactor 값의 비율로 검색 윈도우 크기
	rects = detector.detectMultiScale(gray, scaleFactor=1.05, # 다양한 크기의 얼굴을 검출하기 위하여 처음에는 작은 크기의 검색 윈도우를 이용하여 객체를 검출
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

    # frame위에 감지된 얼굴에 경계 상자를 그리는 과정 
	# 경계상자 반복 
	for (x, y, w, h) in rects:
		# 이미지 위에 얼굴의 경계 상자를 그린다.
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# 프레임의 출력을 보여준다. 
	cv2.imshow("Frame", frame) # frame인자로 지정되고 제목이 Frame인 영상을 보여줌.   
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"): # q를 입력하면 코드 실행 정지
		break

cv2.destroyAllWindows() # 위의 검색 윈도우 청소 or 파괴
vs.stream.release() # 활성된 카메라 끄기 - 실시간 찰영시 주석처리해도 됨
vs.stop()


