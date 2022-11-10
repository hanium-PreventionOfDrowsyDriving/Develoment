#include<SoftwareSerial.h>

#define TRIG 9 //TRIG 핀 설정 (초음파 보내는 핀)
#define ECHO 8 //ECHO 핀 설정 (초음파 받는 핀)
#define BUZ 2 //BUZ 핀 설정 

int count=0;

String str=""; //통신으로 받는 문자열

void setup() {
  Serial.begin(9600); //PC모니터로 센서값을 확인하기위해서 시리얼 통신을 정의해줍니다.              
  pinMode(TRIG, OUTPUT);
  pinMode(ECHO, INPUT);
  pinMode(BUZ, OUTPUT);
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


  // PC모니터로 초음파 거리값을 확인 하는 코드 입니다
  Serial.println(duration ); //초음파가 반사되어 돌아오는 시간을 보여줍니다.
  Serial.print("DIstance : ");
  
  Serial.print(distance); //측정된 물체로부터 거리값(cm값)을 보여줍니다.
  Serial.println(" Cm");
  
  Serial.print("Count : ");
  Serial.println(count);
  Serial.print("\n");
  
  if(distance>15){ 
    //digitalWrite(BUZ,HIGH);
    delay(1000); 
    //digitalWrite(BUZ,LOW);
    count++; //카운터 1회 증가
  }
  
  if(count==3) {    
    Serial.print("Count : ");
    Serial.print(count); //시리얼 모니터에 졸음 운전 신호를 보내기.
    Serial.print("\nWARNING!\n\n");
    digitalWrite(BUZ, HIGH);
    delay(1000); 
    digitalWrite(BUZ, LOW);
    count=0;
  } 
  delay(1000);
}
