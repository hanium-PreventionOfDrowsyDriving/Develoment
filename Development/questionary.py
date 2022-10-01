#-*- coding: utf-8 -*-

import random
from secrets import choice

quizlist = {'lion.mp3':['사자', '하자', '라이언'],
          'cow.mp3':['소', '송아지'],
          'dog.mp3':['개', '강아지', '멍멍이'],
          'duck.mp3':['오리', '우리'],
          'frog.mp3':['개구리'],
          'calculation':['+','-','*']}

class Questionary:
    def __init__(self):
      self.choice = random.choice(list((quizlist.keys())))
      self.answer = quizlist[self.choice]

    #음성 파일 상대경로 출력
    def soundpath(self):
      print("접속완료")
      temp_path = "sound\\"+self.choice
      return temp_path

    def printanswer(self):
      # print(self.answer)
      return self.answer