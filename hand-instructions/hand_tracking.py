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

        # is turtle reversed
        self.isTurtleReversed = False

        # gesture detection on robot
        self.press_down_start = False
        self.press_down_reverse = False

    def findFingers(self, frame, draw=True):
        try:
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except cv2.error as e:
            print(
                "NOTE: This error happens every so often. If you run it again, it should work."
            )
            raise e
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

    def check_gesture(self, gesture):
        finger_one = self.lmsSet[4]  # thumb

        if gesture == "start":
            finger_two = self.lmsSet[8]  # index

        elif gesture == "reverse":
            finger_two = self.lmsSet[20]  # pinky

        distance = calculate_distance(finger_one, finger_two)

        # smoothing the gesture detection for press and release
        if distance < 25:
            if gesture == "start" and not self.press_down_start:
                self.press_down_start = not self.press_down_start
                return False
            elif gesture == "reverse" and not self.press_down_reverse:
                self.press_down_reverse = not self.press_down_reverse
                return False
        else:
            if gesture == "start" and self.press_down_start:
                self.press_down_start = not self.press_down_start
                return True
            elif gesture == "reverse" and self.press_down_reverse:
                self.press_down_reverse = not self.press_down_reverse
                return True
            else:
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


def draw_turtle(frame):
    """Draws the a background rectangle and a circle on the screen to represent the turtle."""

    frame_height, frame_width, _ = frame.shape

    background_rect_height = 125
    background_rect_width = 100
    cv2.rectangle(
        frame,
        (0, frame_height - background_rect_height),
        (background_rect_width, frame_height),
        (208, 253, 255),
        -1,
        lineType=cv2.LINE_AA,
    )

    circle_x = 50
    circle_y = frame_height - 50
    cv2.circle(
        frame,
        (circle_x, circle_y),
        24,
        (0, 171, 102),
        -1,
        lineType=cv2.LINE_AA,
    )


def draw_turtle_arrow(frame, angle_radians, has_arrow=False, is_reversed=False):
    """Draws an arrow on the screen to represent the turtle's direction."""

    frame_height, frame_width, _ = frame.shape

    circle_x = 50
    circle_y = frame_height - 50

    arrow_length = 50

    if is_reversed:
        angle_radians = -angle_radians

    end_point = (
        circle_x + int(arrow_length * math.cos(angle_radians)),
        circle_y + int(arrow_length * math.sin(angle_radians)),
    )

    if has_arrow:
        cv2.arrowedLine(
            frame,
            (circle_x, circle_y),
            end_point,
            (255, 0, 0),
            2,
            line_type=cv2.LINE_AA,
        )


def turtle_direction(frame, angle_radians, has_arrow=False, is_reversed=False):
    frame_height, frame_width, _ = frame.shape
    angle_degrees = math.degrees(angle_radians)

    if has_arrow:
        # this is hardcoded based on the way we calculate the angle
        if angle_degrees < -120 or angle_degrees > 120:
            direction_text = "Left"
        elif 40 <= angle_degrees <= 120:
            direction_text = "Backward"
        elif -60 <= angle_degrees <= 40:
            direction_text = "Right"
        elif -120 <= angle_degrees <= -60:
            direction_text = "Forward"
        else:
            direction_text = "Broken"
    else:
        direction_text = "Stopped"

    cv2.putText(
        frame,
        f"{direction_text}",
        (0, frame_height - 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        lineType=cv2.LINE_AA,
    )

    return direction_text


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

        draw_turtle(frame)
        if lmsSet:  # check if hand is not empty

            # find the angle of the hand gesture
            angle_radians = detector.handAngle()

            # check if the start or reverse gesture is detected
            if detector.check_gesture("start"):
                detector.isTurtleStarted = not detector.isTurtleStarted
            elif detector.check_gesture("reverse"):
                detector.isTurtleReversed = not detector.isTurtleReversed

            # this function draws the arrow and returns the desired direction
            draw_turtle_arrow(
                frame,
                angle_radians,
                detector.isTurtleStarted,
                detector.isTurtleReversed,
            )

            # use direction from here for turtlesim input
            direction = turtle_direction(
                frame,
                angle_radians,
                detector.isTurtleStarted,
                detector.isTurtleReversed,
            )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow("frame", frame)

        # press q to exit the program - it takes a bit to register
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    main()
