import cv2 
import time
import threading


from queue import Queue
from datetime import datetime

from utils.stream import frame_write

'''
cap = cv2.VideoCapture(2, cv2.CAP_V4L2)
codec = cv2.VideoWriter_fourcc('M','J','P','G')
cap.set(6, codec)
cap.set(5, 30)

cap.set(3, 1920)
cap.set(4, 1080)

fps = int(cap.get(5))
print('FPS', fps)
while(cap.isOpened()):
	ret, frame = cap.read()
	frame_time = cap.get(cv2.CAP_PROP_POS_MSEC)
	time_now = datetime.now()
	print(frame.shape, frame_time, time_now)
	
'''


q = Queue(maxsize=500)
count = 0

t = threading.Thread(target=frame_write, args=(q, count))
#t.setDaemon(True)
t.start()

time.sleep(5)
while True:
   if not q.empty():
       #item = cam_thread.queue.get()
       tmp=10
       
       item = q.get()
       
       for i in item:
           print(i[0].shape, i[1], i[2])
           #break
       time.sleep(1.3)
           
   else:
       time.sleep(15)
       #print('q empty')
       

