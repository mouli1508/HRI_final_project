from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
import time

#Set the IP address of your Raspberry Pi
pi_ip = '192.168.1.112'  # Replace with the actual IP address of your Raspberry Pi

#Connect to the Raspberry Pi remotely
pi_factory = PiGPIOFactory(host=pi_ip)

#Create a servo object on pin 23
servo = Servo(23, pin_factory=pi_factory)

try:
    while True:
        # Move the servo to the left
        servo.value = -0.5
        time.sleep(1)

        # Move the servo to the right
        servo.value = 0.25
        time.sleep(1)

except KeyboardInterrupt:
    # Handle keyboard interrupt (Ctrl+C) to stop the servo
    servo.value = 0
    print("\nServo control terminated.")
