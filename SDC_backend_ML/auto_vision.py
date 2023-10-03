from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import cv2

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
cap=cv2.VideoCapture(0)
time.sleep(2.0)
fps = FPS().start()

while True:
    _,frame=cap.read()
    frame=cv2.flip(frame,1)

    x1 = int(0.1*frame.shape[1])
    y1 = int(0.1*frame.shape[0])
    x2 = int(0.9*frame.shape[1])
    y2 = int(0.9*frame.shape[0])
    cv2.rectangle(frame, (x1-1, y1-1), (x2+1, y2+1), (255,0,0) ,3)
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (400, 400))
    cv2.imshow("Frame", frame)

    # grab the frame dimensions and convert it to a blob
    (h, w) = roi.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(roi, (300, 300)),0.007843, (300, 300), 127.5)
    
    # pass the blob through the network and obtain the detections and
	# predictions
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the prediction
        confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the `confidence` is
		# greater than the minimum confidence
        if confidence > 0.2:
			# extract the index of the class label from the
			# `detections`, then compute the (x, y)-coordinates of
			# the bounding box for the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

			# draw the prediction on the frame
            label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# show the output frame
    cv2.imshow("Frame", frame)

    # update the FPS counter
    fps.update()

    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:
        break

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cap.release()
cv2.destroyAllWindows()
