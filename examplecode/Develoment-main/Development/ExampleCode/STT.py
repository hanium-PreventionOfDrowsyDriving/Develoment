from fileinput import filename
import speech_recognition as sr
import pyttsx3
import os
import datetime
import time
import playsound
from gtts import gTTS

def speak(text):
    tts = gTTS(text=text, lang='ko')
    filename='voice.mp3'
    tts.save(filename)
    playsound.playsound(filename)

speak("뭉크뭉크 지혜")

r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source)

try:
    print("you said : " + r.recognize_google(audio, language='ko'))
    speak(r)
except sr.UnknownValueError:
    print("could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
    

