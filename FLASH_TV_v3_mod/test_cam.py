import threading

import numpy as np 
import time as time
from utils.stream import FlashVideoQueueThread

cam_path = '/dev/video0'

cam_thread = FlashVideoQueueThread(cam_path)
stream_params = {'fps':30, 'width':1920, 'height':1080}
#stream_params = {'fps':30, 'width':640, 'height':480}

print('cam param init')
cam_thread.cam_param_intialization(stream_params, apply_codec=True)


frame_grab_interval = 2.0
num_consec_frames = 7 # integer 

'''
while True:
    item = cam_thread.cam_grab_a_frame()
    print(item.frame.shape, item.timestamp, item.count)
    time.sleep(0.25)

'''
print('cam stream start')
t = threading.Thread(target=cam_thread.cam_stream, args=(frame_grab_interval, num_consec_frames))
t.setDaemon(True)
t.start()

#cam_thread.cam_stream(frame_grab_interval, num_consec_frames)

print('cam stream queue start')
time.sleep(5)
while True:
   if not cam_thread.queue.empty():
       item = cam_thread.queue.get()
       #if item.count % num_consec_frames == 0:
       #print(item.frame.shape, item.timestamp, item.count)
       print(item.frame.shape, item.timestamp, item.count % num_consec_frames)

#cam_thread.cam_stop()
