import threading

import numpy as np 
import time as time
from utils.stream import WebcamVideoStream

import cv2

stream = WebcamVideoStream()

stream.start(src='/dev/video0', width=1920, height=1080, fps=30)

i=0
while True:
    img = stream.read()
    cv2.imwrite('frame_%05d.png'%i, img)
    i+=1
    imgr = cv2.resize(img, (640,342))
    cv2.imshow('frames', imgr)
    
    pressedKey = cv2.waitKey(1) & 0xFF
    if pressedKey == ord('q'):
        stop_capture = True
        break
