# cv2: OpenCV library for computer vision.
# mediapipe as mp: MediaPipe library for face mesh detection.
# DigitalInputDevice, Buzzer: GPIO library for Raspberry Pi for handling digital input devices and buzzers.
# serial: Python's serial library for serial communication.
# time: Standard Python library for handling time-related operations.
# datetime: Standard Python library for working with dates and times.
# pynmea2: Library for parsing NMEA sentences from GPS modules.
# RPi.GPIO: GPIO library specific to Raspberry Pi.
# Adafruit_CharLCD: Library for controlling Adafruit character LCD displays.

import cv2
import mediapipe as mp
from gpiozero import DigitalInputDevice, Buzzer
import serial
import time
from datetime import datetime
import pynmea2
import RPi.GPIO as GPIO
import Adafruit_CharLCD as LCD

lcd=LCD.Adafruit_CharLCD(rs=18,en=23,d4=24,d5=25,d6=8,d7=7,cols=16,lines=2)
GPIO.setmode(GPIO.BCM)

# GPIO Pins
VIBRATION_SENSOR_PIN = 16
BUZZER_PIN = 12


# Phone Number to Send SMS
PHONE_NUMBER = "9781961323"

# Initialize Vibration Sensor and Buzzer
vibration_sensor = DigitalInputDevice(VIBRATION_SENSOR_PIN)
buzzer = Buzzer(BUZZER_PIN)

# Initialize GPS Serial Connection
gps_serial = serial.Serial('/dev/ttyS0', baudrate=9600, timeout=1)


def send_sms(message):
    try:
        # Initialize SMS mode
        gps_serial.write(b'AT+CMGF=1\r\n')
        time.sleep(1)
        
        # Set recipient phone number
        gps_serial.write(b'AT+CMGS="' + PHONE_NUMBER.encode('utf-8') + b'"\r\n')
        time.sleep(1)

        # Send SMS content
        gps_serial.write(message.encode('utf-8') + b'\r\n')
        time.sleep(1)
        
        gps_serial.write(b'\x1A')
        time.sleep(1)
        

        print("SMS sent successfully.")
    except Exception as e:
        print(f"Failed to send SMS: {str(e)}")


def get_gps_coordinates():
    latitude, longitude=[0,0]
    # Read response
    response=""
    while response[0:6]!="$GPRMC":
        response = gps_serial.readline().decode('utf-8')
        print(response)
        if response[0:6]=="$GPRMC":
            data=pynmea2.parse(response)
            latitude=data.latitude
            longitude=data.longitude
            
    return latitude, longitude



# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye_landmarks):
    vertical_dist_1 = abs(eye_landmarks[2][1] - eye_landmarks[3][1])
    vertical_dist_2 = abs(eye_landmarks[4][1] - eye_landmarks[5][1])
    horizontal_dist = abs(eye_landmarks[0][0] - eye_landmarks[1][0])
    ear = (vertical_dist_1 + vertical_dist_2) / (2.0 * horizontal_dist)
    return ear

# Function to calculate mouth aspect ratio (MAR)
def mouth_aspect_ratio(mouth_landmarks):
    horizontal_dist = abs(mouth_landmarks[0][0] - mouth_landmarks[1][0])
    vertical_dist1 = abs(mouth_landmarks[2][1] - mouth_landmarks[3][1])
    vertical_dist2 = abs(mouth_landmarks[4][1] - mouth_landmarks[5][1])
    mar = (vertical_dist1+vertical_dist2) / (2.0 * horizontal_dist)
    return mar

# Function to detect drowsiness
def detect_drowsiness(frame, mp_face_mesh, baseline_ear, ear_threshold, baseline_mar, mar_threshold):
    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame and get the face landmarks
    results = mp_face_mesh.process(rgb_frame)

    EarDecision=0
    MouthDecision=0

    right_eye = [33,133,160,144,158,153] # right eye landmark positions
    left_eye = [263,362,387,373,385,380] # left eye landmark positions
    mouth = [61,291,39,181,269,405] # mouth landmark coordinates
    left_eye_landmarks=[]
    right_eye_landmarks=[]
    mouth_landmarks=[]
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract landmarks for left eye, right eye, and mouth
            for pointloc in range(0,6):
                point = face_landmarks.landmark[left_eye[pointloc]]
                left_eye_landmarks.append((point.x * frame.shape[1], point.y * frame.shape[0]))
                
                point = face_landmarks.landmark[right_eye[pointloc]]
                right_eye_landmarks.append((point.x * frame.shape[1], point.y * frame.shape[0]))
                
                point = face_landmarks.landmark[mouth[pointloc]]
                mouth_landmarks.append((point.x * frame.shape[1], point.y * frame.shape[0]))
        
            # Calculate eye aspect ratio (EAR) for both eyes
            ear_left = eye_aspect_ratio(left_eye_landmarks)
            ear_right = eye_aspect_ratio(right_eye_landmarks)

            # Calculate the average eye aspect ratio
            ear_avg = (ear_left + ear_right) / 2.0

            # Calculate mouth aspect ratio (MAR)
            mar = mouth_aspect_ratio(mouth_landmarks)

            # Draw landmarks on the frame
            for landmark in left_eye_landmarks + right_eye_landmarks + mouth_landmarks:
                cv2.circle(frame, (int(landmark[0]), int(landmark[1])), 2, (0, 255, 0), -1)

            # print("EAR Avg: "+ str(ear_avg))
            # print(baseline_ear - ear_threshold)
            # print("MAR Avg: "+ str(mar))
            # print(baseline_mar + mar_threshold)
            

            # Check for drowsiness
            
            if ear_avg < baseline_ear - ear_threshold:
                EarDecision=1
            
            if mar > baseline_mar + mar_threshold:
                MouthDecision=1
                


    return [EarDecision,MouthDecision]



# man Code Starts
lcd.clear()
lcd.message('Drowsines Dtectn\n Alarming Sys')
time.sleep(5)
lcd.clear()
lcd.message('Tracking:')

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh.FaceMesh()


cap = cv2.VideoCapture(0)

# Define baseline EAR and MAR, and thresholds
baseline_ear = 0.27  # Set this value based on the person's baseline EAR when not drowsy
baseline_mar = 0.5  # Set this value based on the person's baseline MAR when not yawning
ear_threshold = 0.05  # Experiment with different values
mar_threshold = 0.1  # Experiment with different values
ECounter=0
MCounter=0

MainSMSCounter=0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    
    # Detect drowsiness
    EyeSec,MthDec = detect_drowsiness(frame, mp_face_mesh, baseline_ear, ear_threshold, baseline_mar, mar_threshold)
    
    # Display result on the frame
    if MthDec==1:
        MCounter+=1
    else:
        MCounter=0
        if EyeSec==1:
            ECounter+=1
        else:
            ECounter=0
    
    
    if ECounter>=20 or MCounter>=20:
        MainSMSCounter+=1
        # Beep the Buzzer
        buzzer.on()
        time.sleep(2)  # Adjust as needed
        buzzer.off()
        cv2.putText(frame, "Drowsy!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        lcd.clear()
        lcd.message('Tracking:\nDrowsines Detect')
        time.sleep(2)  # Adjust as needed
        ECounter=0
        MCounter=0
    else:
        lcd.clear()
        lcd.message('Tracking:\nNormal')
        

    cv2.imshow('Drowsiness Detection', frame)
    
    if MainSMSCounter>5:
        # Beep the Buzzer
        buzzer.on()
        time.sleep(2)  # Adjust as needed
        buzzer.off()
        print("High Drowsiness Detected Stop your Car")
        lcd.clear()
        lcd.message('High Drowsiness\n   Detected')
        # Get GPS Coordinates
        latitude, longitude = get_gps_coordinates()
        print(f"GPS Coordinates: Latitude {latitude}, Longitude {longitude}")

        # Send SMS
        sms_message = f"Drowsiness Exceeded at {datetime.now()} - GPS Coordinates: {latitude}, {longitude}"
        send_sms(sms_message)

        
        # Wait for some time to avoid repeated readings due to vibrations
        time.sleep(10)
        MainSMSCounter=0
    

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
    
    if vibration_sensor.value:
            print("Accident detected!")
            lcd.clear()
            lcd.message('Accident Occur')
            # Get GPS Coordinates
            latitude, longitude = get_gps_coordinates()
            print(f"GPS Coordinates: Latitude {latitude}, Longitude {longitude}")

            # Send SMS
            sms_message = f"Accident detected at {datetime.now()} - GPS Coordinates: {latitude}, {longitude}"
            send_sms(sms_message)

            # Beep the Buzzer
            buzzer.on()
            time.sleep(2)  # Adjust as needed
            buzzer.off()
            # Wait for some time to avoid repeated readings due to vibrations
            time.sleep(10)
            

# Release resources
cap.release()
cv2.destroyAllWindows()