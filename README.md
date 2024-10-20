# garbanzo

Attempting to build an AI &amp; Computer Vision Powered Robot. This is for the University of Helsinki Computer Science Project Course.

Our unifying theme is using AI to steer robots, and we intend to have 2-3 prototypes

1. Hand-tracking instructions
   Help from:
   https://lvimuth.medium.com/hand-detection-in-python-using-opencv-and-mediapipe-30c7b54f5ff4

![alt text](image.png)

Instructions: (ask Bennett for help to set up)
pip3 install requirements.txt
or
pip3 install opencv-python
brew install open-cv

run with
`python3 hand-instructions/hand_tracking.py`

tapping thumb with index finger will start/stop the robot.

use a single hand to steer, and you will see it represented on the screen

2. NLP instructions
   from a human to LLM
   Given human readable instructions to LLM, LLM generates instructions for the robot.

3. LLM instructions

LLM steers the robot itself, however it wants, given a goal
