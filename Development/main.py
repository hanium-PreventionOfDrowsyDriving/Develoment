#-*- coding: utf-8 -*-

from statistics import quantiles
import playsound
from symbol import break_stmt
import voice_processing
import questionary
import speech_recognition as sr

notice = voice_processing.Notice()
# stt = voice_processing.Stt()
quiz = questionary.Questionary()

# 초기 시작 문구 출력
playsound.playsound('sound\warning.mp3')
notice.start()

#질문 랜덤 출력
try:
  playsound.playsound(quiz.soundpath())
except:
  while True:
    playsound.playsound(quiz.soundpath())

recognizer = sr.Recognizer()
mic = sr.Microphone()

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
    notice.incorrect()