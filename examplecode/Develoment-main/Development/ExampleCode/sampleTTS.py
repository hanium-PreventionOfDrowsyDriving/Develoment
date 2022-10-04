import speech_recognition as sr
from gtts import gTTS
import os


Recognizer = sr.Recognizer() #인스턴스 생성
mic = sr.Microphone()
with mic as source: #안녕~이라고 말하면
    audio = Recognizer.listen(source)
try:
    data = Recognizer.recognize_google(audio, language="ko")
except:
    print("이해하지 못했음")
    
print(data)

def speak(text ,lang="ko", speed=False):
    tts = gTTS(text=text, lang=lang , slow=speed)
    tts.save("./tts.mp3") #tts.mp3로 저장
    os.system("afplay " + "./tts.mp3") #말하기

speak("안녕 나는 뭉크지혜", "ko")