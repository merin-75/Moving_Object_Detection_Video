import cv2

# Load Haar Cascade correctly
alg = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

video_path = r"c:\Users\merin philip\Downloads\855564-hd_1920_1080_24fps.mp4"
cam = cv2.VideoCapture(video_path)

while True:
    ret, img = cam.read()
    #if not ret:
     #   break

    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = haar_cascade.detectMultiScale(grayImg, 1.3, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        img = cv2.resize(img, (600, 350))

    cv2.imshow("Face Detection", img)

    if cv2.waitKey(10) == 27:  # ESC key
        break

cam.release()
cv2.destroyAllWindows()


