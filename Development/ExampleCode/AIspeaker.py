# from fileinput import filename
# from gtts import gTTS
# import os
# import speech_recognition as sr
# import playsound

# def speak(text , a):
#     tts = gTTS(text=text, lang='ko')
#     filename = 'C://' + text + str(a) + '.mp4'
#     tts.save("filename")
#     playsound.playsound(filename)

# Recognizer = sr.Recognizer()
# mic = sr.Microphone()

# i = 0
# list01 = []

# while True:
#     with mic as source:
#         audio = Recognizer.listen(source)
#     try:
#         data = Recognizer.recognize_google(audio ,language="ko")
#     except:
#         speak("이해하지 못하는 말이에요")
#     print(data)
#     if "시리" in data or "시뤼" in data or "시리야" in data:
#         speak("네", i)
#         i = i + 1
#         print("에리스 : 넹")
#     else:
#         speak("다시 불러주세요", i)
#         i = i + 1

#     list01.append(str(i-1) + '.mp3')

# for item in list01:
#     print(item)
    