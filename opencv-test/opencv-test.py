import cv2

image = cv2.imread("/Users/bennettsummy/desktop/IMG_2474.jpeg")
if image is None:
    raise FileNotFoundError("The image file was not found at the specified path.")

# face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = face_cascade.detectMultiScale(image, scaleFactor=1.3, minNeighbors=5)

for x, y, w, h in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
