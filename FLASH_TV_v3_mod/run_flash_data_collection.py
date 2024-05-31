import sys
import os 

# import queue libraries
import subprocess
import threading as th
import traceback
from queue import Queue

# time libs
import time
from datetime import datetime 

# computer vision libs
import cv2
import numpy as np

# custom libs
from flash_runtime_utils import check_face_presence, cam_id
from flash_main import FLASHtv  

def frame_write(q, frm_count):
    idx = cam_id()
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    #cap = cv2.VideoCapture('/dev/video'+str(idx), cv2.CAP_V4L2)
    codec = cv2.VideoWriter_fourcc('M','J','P','G')
    cap.set(6, codec)
    cap.set(5, 30)

    cap.set(3, 1920)
    cap.set(4, 1080)

    fps = int(cap.get(5))
    #print('fps: ', fps)

    count = frm_count
    write_img = True

    global stop_capture
    stop_capture = False
    
    t_st = time.time()
    timer_sec = 0
    last_frame_time = None
    fps_count = 1
    data_list = []
    
    while(cap.isOpened() and not stop_capture): 
        ret, frame = cap.read()
        frame_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        time_now = datetime.now()
        
        if last_frame_time is not None:
            timer_sec = timer_sec + (frame_time - last_frame_time)
            fps_count += 1

        if not ret:
            break
            
        if timer_sec >= 1940 and timer_sec <=2060:
            write_img = not write_img
            timer_sec = 0

        
        if write_img:
            data_list.append([frame, count, time_now])
            #cv2.imwrite(os.path.join(frm_write_path, str(count).zfill(6)+'.png')
        
        if len(data_list)==7:
            a = q.put(data_list)
            write_img = not write_img
            data_list = []
            
        count += 1
        
        if (count+1)%1000 == 0:
            tmp = 10
            print('time for capturing 1000 images:::: ', time.time()-t_st)
            t_st = time.time()
            #break
        
        last_frame_time = frame_time
    
    print('The cam for batch processing is stopped.')
    cap.release()
    time.sleep(3)
    cv2.destroyAllWindows()



# super variables 
write_image_data = True
frames_path = '/home/flashsys008/dmdm2023/data/tmp_frames'
log_file = '/home/flashsys008/dmdm2023/data/tmp.txt'


flash_tv = FLASHtv(family_id='123', data_path='/home/flashsys008/dmdm2023/data', frame_res_hw=None, output_res_hw=None)

rotate_to_find_tc = False

while True:
    
    # CHECK if the face is present to start FLASH
    tc_presence_duration, log_file = check_face_presence(log_file, )
    #frm_counter = log_file[1]
    log_fname = log_file #[0]
    
        
    # INITIATE the frame capture queue 
    frame_counter = 0
    print('starting the batch cam', frame_counter)
    q = Queue(maxsize=500)
    stop_capture = False
    p1 = th.Thread(target=frame_write, args=(q, frame_counter,))
    p1.start()
    time.sleep(5)


    # PROCESS the QUEUE
    batch_data = []
    batch_count = 0
    time_batch_start = time.time()
    batch_write = True
    qempty_start = None
    
    while True:
        if (batch_count+1) % 100 == 0: # to capture the time for frame capture
            if batch_write:
                print('############################################################')
                print('Time for processing 100 batch7s: ', time.time()-t_batch_start)
                print('############################################################')
                time_batch_start = time.time()
            batch_write = False    

        if not q.empty():
            qempty_start = None
            batch_data = q.get() # image frames, counter, time-stamps
        
            batch_count+=1 
            batch_write = True
            
            frame_1080p_ls = [b[0] for b in batch_data]
            frame_counts = [b[1] for b in batch_data]
            frame_stamps = [b[2] for b in batch_data]
        
            frm_counter = frame_counts[-1]
            tdet = time.time()
            if write_image_data:
                tmp = [cv2.imwrite(os.path.join(frames_path, str(frame_counts[k]).zfill(6)+'.png'), frame_1080p_ls[k]) for k in range(3,5)]
                del tmp        
            
            frame_1080p_ls = [cv2.cvtColor(img1080, cv2.COLOR_BGR2RGB) for img1080 in frame_1080p_ls]
            frame_608p_ls = [cv2.resize(img1080, (608,342)) for img1080 in frame_1080p_ls]
            
            frame_1080p_ls = [frame_1080p_ls[3],frame_1080p_ls[4]]
            frame_608p_ls = [frame_608p_ls[3],frame_608p_ls[4]]
            
            # Analyze the set of frames
            # detect the child in the frames
            if rotate_to_find_tc:
                tmp=10
            else:
                frame_bls = [flash_tv.run_detector(img[:,:,::-1]) for img in frame_1080p_ls]
            print(len(frame_bls[0]), len(frame_bls[1]))
            
            # perform gaze estimation if the child is there
            
        
        

