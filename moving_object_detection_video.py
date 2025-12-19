import cv2
import imutils

# Load video file (put video in same folder)
video = cv2.VideoCapture(r"c:\Users\merin philip\Downloads\Media_vtest.avi")

firstFrame = None
area = 300

while True:
    ret, frame = video.read()
    if not ret:
        break

    text = "Normal"

    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if firstFrame is None:
        firstFrame = gray
        continue

    frameDelta = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    motion_detected = False

    for c in cnts:
        if cv2.contourArea(c) < area:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_detected = True

    if motion_detected:
        text = "Moving Object Detected"

    cv2.putText(frame, text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Moving Object Detection - Video", frame)

    key = cv2.waitKey(30)
    if key == ord("q"):
        break

    # Update background slowly
    firstFrame = gray

video.release()
cv2.destroyAllWindows()
