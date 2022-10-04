from fileinput import filename
from gtts import gTTS
import playsound
import os
import speech_recognition as sr
import time

def speak(text):
    tts = gTTS(text=text, lang='ko')
    filename = "vic.mp3"
    tts.save(filename)
    playsound.playsound(filename)

def reset():
    os.remove(r"vic.mp3")

class Questionary:
    '''
    졸음운전 질문지
    '''
    def __init__(self, name, informations):
        self.name = name
        self.informations = informations
        
    def aboutQuest(self):
        res = "이 동물은" + self.name + "이며, " + self.informations + "입니다."
        return res

animal = Questionary('사자', '육식동물')
result = animal.aboutQuest()

playsound.playsound('lion.mp3')
speak("정답을 얘기해주세요")
reset()
Recognizer = sr.Recognizer()
mic = sr.Microphone()

while True:
    with mic as source:
        audio = Recognizer.listen(source)
    try:
        data = Recognizer.recognize_google(audio, language="ko")
    except:
        speak("이해를 못했어요")
        reset()
    print(data)

    if "사자" in data or "라이언" in data:
        speak("정답입니다")
        break
    else:
        speak("틀렸습니다, 다시 한번 얘기해주세요")
        reset()