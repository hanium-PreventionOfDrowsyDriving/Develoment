# python detect_face_parts.py --shape-predictor models/shape_predictor_68_face_landmarks.dat --image images/Face2.jpg

from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
# 파라메터 구문 분석
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# 얼굴 감지에 사용할 dlib 얼굴 랜드마크 감지 파일 경로
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
# load the input image, resize it, and convert it to grayscale
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# grayscale image에서 얼굴 감지
rects = detector(gray, 1)


# 얼굴 감지 반복
for (i, rect) in enumerate(rects):
	# 얼굴영역에 랜드마크를 결정
	# Numpy로 얼굴 랜드마크(x,y) 좌표로 변환
	shape = predictor(gray, rect)
	shape = face_utils.shape_to_np(shape)
	# 얼굴 특정 부위 감지 반복
	for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
		
		# 랜드마크 그리기 위해 원본 이미지를 클론
		# 이미지 위에 얼굴 특정 부분 이음을 표시하기 위해 디스플레이
		clone = image.copy()
		cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
			0.7, (0, 0, 255), 2)
		
		# 특정 얼굴 부분에 랜드마크 그리기 반복
		for (x, y) in shape[i:j]:
			cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)


# extract the ROI of the face region as a separate image
		(x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
		roi = image[y:y + h, x:x + w]
		roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
		# show the particular face part
		cv2.imshow("ROI", roi)
		cv2.imshow("Image", clone)
		cv2.waitKey(0)
	# visualize all facial landmarks with a transparent overlay
	output = face_utils.visualize_facial_landmarks(image, shape)
	cv2.imshow("Image", output)
	cv2.waitKey(0)
