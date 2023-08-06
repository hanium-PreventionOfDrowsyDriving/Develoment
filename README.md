
# 1. 가상환경 설치 가이드

<a href="https://blog.naver.com/PostView.nhn?blogId=rhrkdfus&logNo=221369959311">가상환경 설치</a>
<br>
<a href="https://velog.io/@moey920/virtualenv%EB%A5%BC-%ED%99%9C%EC%9A%A9%ED%95%9C-%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EA%B0%80%EC%83%81%ED%99%98%EA%B2%BD-%EC%84%A4%EC%A0%95">가상환경 명령어</a>

## workon error
if error : bash: command no found <br> 
참고 사이트: https://github.com/davidtavarez/pwndb/issues/28 <br>
-> source ~/.profile <br>
-> $ virtualenv venv

# 2. 필요 라이브러리 설치
## 2.1. opencv install guide
<a href="https://pyimagesearch.com/2018/09/19/pip-install-opencv/">opencv 설치</a>

## opencv 설치 중 오류 해결
<a href="https://supersfel.tistory.com/257?category=1057215">opencv 설치 오류 해결</a>
오류 종류: <br>
- sudo apt install libgtk-3-dev libqtgui4 libqtwebkit4 libqt4-test python3-pyqt5 error

## 2.2. install dlib
<a href="https://pyimagesearch.com/2017/05/01/install-dlib-raspberry-pi/">dlib 설치</a>

# 3. 가상환경 가이드
// 가상환경 사용 필요 시 참고 

## 만들어둔 가상환경 실행(CMD에서)
- 가상환경 실행:  workon <가상환경 이름>
<br>
e.g. workon cv

<br>

### if occur error 
- bash: workon: command not found 
1 $ source ~/.profile
2 $ virtualenv venv
3 $ workon cv

### cmd창 디렉토리 변경 -> 파이썬 코드 파일이 다른 디렉토리에 저정되어있기에
1 $ cd Desktop/Hanium/project
2 $ pwd

<hr>

# 4. 파일 실행
python main.py --haarcascade haarcascade_frontalface_default.xml --shape-predictor shape_predictor_68_face_landmarks.dat

# 5. 파일 설명

- main.py: 통합 파일(항상 최종본 유지)

(1) 고개 각도 
- ArduinoSerial.py: 초음파 관련 통신 모듈, main.py 에서 import 해서 사용

(2) 눈깜빡임 인식
- haarcascade_frontalface_defaul: 얼굴인식 모듈
- shape_predictor_68_face_landmarks.dat: 얼굴인식 모듈

(3) 차선 인식
- cars.xml: 차선 인식 모듈

(4) STT&TTS 
- main_STTTTS.py: 대화 기능 메인 파일
- MDL_questionary.py: 퀴즈 관련 모듈
- MDL_voice_processing.py: 음성 처리 관련 모듈
- tempCodeRunnerFile.py
