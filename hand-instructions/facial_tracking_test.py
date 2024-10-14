import cv2

# print(cv2.data.haarcascades)
hand_cascade = cv2.CascadeClassifier( cv2.data.haarcascades + 'harrcascade_palm.xml')


cap = cv2.VideoCapture(0)  # Use 0 for the default webcam

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect hands
    hands = hand_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected hands
    for (x, y, w, h) in hands:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Hand Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
