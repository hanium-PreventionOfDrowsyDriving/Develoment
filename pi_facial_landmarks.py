# python pi_facial_landmarks.py

from imutils import face_utils
import dlib
import cv2
# 68.facelandmark 모델 경로
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
# 이미지 경로 및 회색조로 변경
image = cv2.imread("images/Face1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# grayscale image 로 얼굴감지
rects = detector(gray, 0)
# 얼굴 감지 반복
for (i, rect) in enumerate(rects):
	# 랜드마크 얼굴 감지 및 numpy로 얼굴 객체 엑스 와이 좌표 찍기
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	# 랜드마크를 찍기 위해 엑스 와이 좌표 연속 찍기
	# 그리고 이미지 위에 그림
	for (x, y) in shape:
		cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
# 얼굴 인식 및 랜드마크  출력 및 보여주기
cv2.imshow("Output", image)
cv2.waitKey(0)