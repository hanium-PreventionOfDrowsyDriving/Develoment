#include<SoftwareSerial.h>

#define TRIG 9 //TRIG 핀 설정 (초음파 보내는 핀)-+
#define ECHO 8 //ECHO 핀 설정 (초음파 받는 핀)
//#define LED 2 //LED 핀 설정 (빨강 LED 핀)-->부저로 교체하기 전, 일단 LED

int count=0;

String str=""; //통신으로 받는 문자열

void setup() {
  Serial.begin(9600); //PC모니터로 센서값을 확인하기위해서 시리얼 통신을 정의해줍니다.              
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
}

void loop(){
  
  long duration, distance;

  digitalWrite(TRIG, LOW); //초기화
  delayMicroseconds(2);
  digitalWrite(TRIG, HIGH); //Trigger 신호 발생
  delayMicroseconds(10);
  digitalWrite(TRIG, LOW);

  duration = pulseIn (ECHO, HIGH); //Echo 신호 입력
  distance = duration * 17 / 1000; //거리계산


  // PC모니터로 초음파 거리값을 확인 하는 코드 입니다.

  //Serial.println(duration ); //초음파가 반사되어 돌아오는 시간을 보여줍니다.
  //Serial.print("\nDIstance : ");
  //Serial.print("\n");
  //Serial.print(distance); //측정된 물체로부터 거리값(cm값)을 보여줍니다.
  //Serial.println(" Cm");
  //Serial.print("Count : ");

  //Serial.println(count);
  
  if(distance<15){ //15cm 이하일 경우 빨간 LED 출력 
    //digitalWrite(LED,HIGH);
    delay(500); //2초후에 다시 측정
    count++; //카운터 1회 증가
    //Serial.println(count); //시리얼 모니터에 졸음 운전 신호를 보내기.
  }
  
  if(count==3) { 
    //카운트가 3회 누적 시, 졸음운전으로 판단하여 라즈베리파이에 블루투스로 졸음 운전 신호를 보낸다.      
    Serial.print(count); //시리얼 모니터에 졸음 운전 신호를 보내기.
    //mySerial.println(sstop); //여기에 블루투스로 졸음 운전 신호를 보내기.
    //digitalWrite(LED, HIGH);
    count=0;

    // digitalWrite(LED, LOW);
    //시스템 초기화 ex) 몇초의 텀을 두고, count를 0으로 초기화시키기 또는 라즈베리파이에서 신호를 다시 받을 때까지 대기(라즈베리파이에서 신호를 받으면 count를 0으로 초기화하게 해도 될 듯
  } 
  
  //else //15cm 이상일 경우 빨간 LED 끔
  // digitalWrite(LED,LOW)
  delay(1000);
  
}
