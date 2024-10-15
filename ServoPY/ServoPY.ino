#include <Servo.h>
Servo myservo;  // create servo object to control a servo

const int ledPinOn = 13;  // the pin that the LED is attached to
const int ledPinOff = 12;  // the pin that the LED is attached to
const int ServoPin = 11;  // the pin that the LED is attached to



int pos = 0;    // variable to store the servo position
int limit = 0;
int stepper = 1;
String incomingByte;       // a variable to read incoming serial data into
int oldByte = 0;

void setup() {
  // initialize serial communication:
  Serial.begin(9600);
  // initialize the LED and servo pin as an output:
  pinMode(ledPinOn, OUTPUT);
  pinMode(ledPinOff, OUTPUT);
  myservo.attach(ServoPin);  
}

void loop() {
  // see if there's incoming serial data:
  if (Serial.available()) {

    digitalWrite(ledPinOn, HIGH);
    digitalWrite(ledPinOff, LOW);
    // read the oldest byte in the serial buffer:
    incomingByte = Serial.readString(); // read data until newline
    limit = incomingByte.toInt();   // change datatype from string to integer   
   
   
    if (limit<=oldByte)
            {for (pos = oldByte; pos >= 0; pos -= 1) { // goes from 180 degrees to 0 degrees
            myservo.write(pos);              // tell servo to go to position in variable 'pos'
            delay(15);                       // waits 15ms for the servo to reach the position
          }
          }
    
     else {for (pos = oldByte; pos <= limit; pos += 1) { // goes from 0 degrees to 180 degrees
            // in steps of 1 degree
            myservo.write(pos);              // tell servo to go to position in variable 'pos'
            delay(15);                       // waits 15ms for the servo to reach the position
          }
          }





    Serial.print(oldByte);  
    Serial.println(incomingByte);
    digitalWrite(ledPinOn, LOW);
    digitalWrite(ledPinOff, HIGH);
    oldByte = pos;

      
  }
}