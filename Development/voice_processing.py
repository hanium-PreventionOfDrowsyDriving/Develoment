#-*- coding: utf-8 -*-

from fileinput import filename
from multiprocessing import reduction
import speech_recognition as sr
from gtts import gTTS
import playsound
import os

class Playback:
  def tts(self, text):
    print("접속 완료")

    g_tts = gTTS(text=text, lang='ko')
    filename = "vic.mp3"
    g_tts.save(filename)
    playsound.playsound(filename)
    os.remove(filename)

  def reset():
    os.remove(r"vic.mp3")
    
class Notice(Playback):
  def __init__(self) :
    self.starting = "졸음이 감지되었습니다. 운전자님의 졸음운전 예방을 위한 퀴즈가 진행됩니다. 3회 오답 시 경고음이 울립니다. 답변 시간은 3초 입니다."
    self.tinputreq = "정답을 말씀해주세요"
    self.tincorrect = "틀렸습니다"
    self.tcorrect = "정답입니다 주행 모드로 전환합니다"
    self.tinput_error = "답변을 이해하지 못했습니다 다시 한 번 이야기해주세요"
    
  # 시작문구
  def start(self):
    super().tts(self.starting)

  # 입력요청 문구
  def inputreq(self):
    super().tts(self.tinputreq)

  # 정답문구
  def correct(self):
    super().tts(self.tcorrect)

  # 오답문구
  def incorrect(self):
    super().tts(self.tincorrect)

  # 입력오류 문구
  def inputerror(self):
    super().tts(self.tinput_error)


# class Stt():
#   def __init__(self):
#     self.recognizer = sr.Recognizer()
#     self.mic = sr.Microphone()

#   def recording(self):
#     with self.mic as source:
#       audio = self.recognizer.listen(source)
#     return audio

#   def recognizer(self):
#     return self.recognizer