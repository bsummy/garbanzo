import numpy as np
import cv2
import mediapipe as mp
import time
import math as math


class HandTrackingDynamic:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.__mode__ = mode
        self.__maxHands__ = maxHands
        self.__detectionCon__ = detectionCon
        self.__trackCon__ = trackCon
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

        # is turtle started
        self.isTurtleStarted = False

        # gesture detection on robot
        self.press_down = False

    def findFingers(self, frame, draw=True):
        # print(frame)
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(
                        frame, handLms, self.handsMp.HAND_CONNECTIONS
                    )

        return frame

    def findPosition(self, frame, handNo=0, draw=True):
        xList = []
        yList = []
        bbox = []
        self.lmsSet = {}
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmsSet[id] = (cx, cy)
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            # print("Hands Keypoint")
            # print(bbox)
            if draw:
                cv2.rectangle(
                    frame,
                    (xmin - 20, ymin - 20),
                    (xmax + 20, ymax + 20),
                    (0, 255, 0),
                    2,
                )

        return self.lmsSet, bbox

    def findFingerUp(self):
        fingers = []
        """
        0 is the wrist
        1-4 are the thumb
        5-8 are the index finger
        9-12 are the middle finger
        13-16 are the ring finger
        17-21 are the pinky finger

        if 4
        """
        # check if the hand is left or right

        isRightHand = self.lmsSet[0][0] < self.lmsSet[4][0]
        # thumb - checking the x-value of the tip of the thumb and the base of the thumb

        thumb_base = self.lmsSet[1][0]
        thumb_tip = self.lmsSet[4][0]

        thumb_up_result = (
            (thumb_tip > thumb_base) if isRightHand else (thumb_tip < thumb_base)
        )
        fingers.append(thumb_up_result)

        # checking if the other fingers are up (y-values)
        for id in range(5, 21, 4):
            fingers.append(self.lmsSet[id][1] > self.lmsSet[id + 3][1])

        return fingers

    def findDistance(self, p1, p2, frame, draw=True, r=15, t=3):
        x1, y1 = self.lmsSet[p1][1:]
        x2, y2 = self.lmsSet[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(frame, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (255, 0, 0), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 0.255), cv2.FILLED)
        len = math.hypot(x2 - x1, y2 - y1)

        return len, frame, [x1, y1, x2, y2, cx, cy]

    def handAngle(self):
        wrist = self.lmsSet[0]
        middle = self.lmsSet[12]

        angle = calculate_hand_angle(wrist, middle)
        return angle

    def check_gesture(self):
        thumb_tip = self.lmsSet[4]
        index_tip = self.lmsSet[8]

        distance = calculate_distance(thumb_tip, index_tip)

        if distance < 25 and not self.press_down:
            self.press_down = True
        elif distance > 25 and self.press_down:
            self.press_down = False
            return True
        return False


def calculate_distance(point1, point2):
    """Calculates the Euclidean distance between two points.

    Args:
        point1: A tuple representing the coordinates of the first point (x1, y1).
        point2: A tuple representing the coordinates of the second point (x2, y2).

    Returns:
        The Euclidean distance between the two points.
    """

    x1, y1 = point1
    x2, y2 = point2

    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    return distance


def draw_turtle(frame, angle_radians, has_arrow=False):
    rectangle_width = 100
    rectangle_height = 50
    arrow_length = 50

    frame_height, frame_width, _ = frame.shape
    x = (frame_width - rectangle_width) // 2
    y = frame_height - rectangle_height
    cv2.circle(
        frame,
        (frame_width // 2, y),
        min(rectangle_width, rectangle_height) // 2,
        (0, 171, 102),
        -1,
    )

    end_point = (
        x + rectangle_height + int(arrow_length * math.cos(angle_radians)),
        y + int(arrow_length * math.sin(angle_radians)),
    )

    if has_arrow:
        cv2.arrowedLine(frame, (x + rectangle_height, y), end_point, (255, 0, 0), 2)


def calculate_hand_angle(circle_center, hand_center):
    """Calculates the angle of the hand relative to a 360-degree circle.

    Args:
      circle_center: Tuple representing the (x, y) coordinates of the circle's center.
      hand_center: Tuple representing the (x, y) coordinates of the hand's center.

    Returns:
      The angle of the hand relative to the circle, in degrees.
    """

    # Calculate the vector from the circle's center to the hand's center
    vector_x = hand_center[0] - circle_center[0]
    vector_y = hand_center[1] - circle_center[1]

    # Calculate the angle in radians
    angle_radians = math.atan2(vector_y, vector_x)

    # # Convert the angle to degrees
    # angle_degrees = math.degrees(angle_radians)

    # # Adjust the angle if needed based on your 360-degree circle's orientation
    # # For example, if the circle starts at the top and goes clockwise:
    # angle_degrees -= 90

    # angle_radians += math.pi / 2
    return angle_radians


def main():
    cap = cv2.VideoCapture(0)
    detector = HandTrackingDynamic()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()

        frame = detector.findFingers(frame)
        lmsSet, bbox = detector.findPosition(frame, draw=True)
        if lmsSet:  # check if dict not empty

            # find the angle of the fist gesture
            angle_radians = detector.handAngle()

            # figure this out later - need to make sure each gesture is only detected once
            if detector.check_gesture():
                detector.isTurtleStarted = not detector.isTurtleStarted

            draw_turtle(frame, angle_radians, detector.isTurtleStarted)

            # cv2.putText(
            #     frame,
            #     "On: " + str(isTurtleStarted),
            #     (10, 70),
            #     cv2.FONT_HERSHEY_PLAIN,
            #     3,
            #     (255, 0, 255),
            #     3,
            # )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("frame", frame)

        # press q to exit the program - it takes a bit to register
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
