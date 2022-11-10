#-*- coding: utf-8 -*-

from statistics import quantiles
import playsound
from multiprocessing import Process
from symbol import break_stmt
import questionary
import voice_processing
import speech_recognition as sr
import os
import sys
  
notice = voice_processing.Notice()
# stt = voice_processing.Stt()
quiz = questionary.Questionary()


# 초기 시작 문구 출력
notice.start()

path = 'sound/'
playsound.playsound(path+'warning.mp3')

#질문 랜덤 출력
playsound.playsound(quiz.soundpath())


recognizer = sr.Recognizer()
m = sr.Microphone()

notice.inputreq()
while True:
    # audio = stt.recording()
    with sr.Microphone() as source:
        print('1')
        audio = sr.Recognizer().listen(source)
    try:
        print('2')
        data = recognizer.recognize_google(audio, language="ko")
        input = str(data)
        print(input)
    except:
        print('a')
        notice.inputerror()
    
    answer = quiz.printanswer()
    
    end = False
    
    for aword in answer:
        if aword in input:
            notice.correct()
            end = True
            break
        
    if end == True:
        break
    else:
        notice.incorrect()
