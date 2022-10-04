#-*- coding: utf-8 -*-

from statistics import quantiles
import playsound
from multiprocessing import Process
from symbol import break_stmt
import MDL_questionary
import MDL_voice_processing
import speech_recognition as sr
import os
import sys

  
notice = MDL_voice_processing.Notice()
# stt = voice_processing.Stt()
quiz = MDL_questionary.Questionary()

try:
  play = Process(target = playsound, args=(quiz.soundpath(),))
  play.start()
  play.terminate()
except:
  os.execl(sys.executable, sys.executable, *sys.argv)

# 초기 시작 문구 출력
path = 'sound/'
#playsound.playsound(path+'warning.mp3')
play = Process(target = playsound, args=(quiz.soundpath(),))
play.start()
play.terminate()
notice.start()

#질문 랜덤 출력
playsound.playsound(quiz.soundpath())

recognizer = sr.Recognizer()
mic = sr.Microphone()
'''
notice.inputreq()
while True:
  # audio = stt.recording()
  with mic as source:
      audio = recognizer.listen(source)
  try:
    data = recognizer.recognize_google(audio, language="ko")
    input = str(data)
  except:
    notice.inputerror()

  print(input)

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
    notice.incorrect()'''