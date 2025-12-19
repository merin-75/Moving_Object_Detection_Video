# Importing libraries
import cv2
import imutils

# Loading video
video = cv2.VideoCapture(
    r"C:\Users\merin philip\Downloads\traffic.mp4"
)

# Variables
firstFrame = None
area = 300

# Loop start
while True:
    ret, frame = video.read()
    if not ret:
        break

    text = "Normal"

    # Resize frame
    frame = imutils.resize(frame, width=500)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Blur image
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Set background frame
    if firstFrame is None:
        firstFrame = gray
        continue

    # Frame difference
    frameDelta = cv2.absdiff(firstFrame, gray)

    # Threshold
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

    # Dilate
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cnts = imutils.grab_contours(cnts)

    motion_detected = False

    # Loop through contours
    for c in cnts:
        if cv2.contourArea(c) < area:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        motion_detected = True

    # Update text
    if motion_detected:
        text = "Moving Object Detected"

    # Display text
    cv2.putText(
        frame, text, (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
    )

    # Show output
    cv2.imshow("Moving Object Detection - Video", frame)

    # Key press
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break

    # Update background slowly
    firstFrame = gray

# Release resources
video.release()
cv2.destroyAllWindows()



