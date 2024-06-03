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
from flash_main import FLASHtv  

from utils.flash_runtime_utils import check_face_presence, cam_id, write_log_file, correct_rotation
from utils.visualizer import draw_rect_ver, draw_gz

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
rotate_to_find_tc = False

frames_path = '/home/'+os.getlogin()+'/dmdm2023/data/tmp_frames'
frames_save_path = '/home/'+os.getlogin()+'/dmdm2023/data/tmp_frames_res'
log_path = '/home/'+os.getlogin()+'/dmdm2023/data/tmp.txt'
frame_counter = 1

flash_tv = FLASHtv(family_id='123', data_path='/home/'+os.getlogin()+'/dmdm2023/data', frame_res_hw=None, output_res_hw=None)
log_file = [log_path, frame_counter]

# TO ADD
 # add rotate to find TC
 # add args for variables
 # add frame input send
 
# TO TEST
 # real time 
 

while True:
    print('*************************************************************************************')
    print('INITIALIZING THE DETECTOR MODEL; 5 mins time to load')
    print('*************************************************************************************')
    
    # CHECK if the face is present to start FLASH
    tc_presence_duration, log_file = check_face_presence(log_file, flash_tv.run_detector)
    frame_counter = log_file[1] 
    log_path = log_file[0]
    face_seen_last_time = datetime.now()
        
    # INITIATE the frame capture queue 
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
    
    try:
        log_lines = []
        while True:
            if (batch_count+1) % 100 == 0: # to capture the time for frame capture
                if batch_write:
                    print('############################################################')
                    print('Time for processing 100 batch7s: ', time.time()-time_batch_start)
                    print('############################################################')
                    time_batch_start = time.time()
                batch_write = False    
            
            # if no face is detected for 1500 secs () duration then fall back to stand by mode    
            stand_by_mode_check =  datetime.now() - face_seen_last_time
            if stand_by_mode_check.total_seconds() >= 1500:
                stop_capture = True
                time.sleep(5)
                p1.join()
                del q
                break
            
            if not q.empty():
                qempty_start = None
                batch_data = q.get() # image frames, counter, time-stamps
            
                batch_count+=1 
                batch_write = True
                
                frame_1080p_ls = [b[0] for b in batch_data]
                frame_counts = [b[1] for b in batch_data]
                frame_stamps = [b[2] for b in batch_data]
            
                frame_counter = frame_counts[-1]
                tdet = time.time()
                if write_image_data:
                    tmp = [cv2.imwrite(os.path.join(frames_path, str(frame_counts[k]).zfill(6)+'.png'), frame_1080p_ls[k]) for k in range(3,5)]
                    del tmp        
                
                frame_1080p_ls = [cv2.cvtColor(img1080, cv2.COLOR_BGR2RGB) for img1080 in frame_1080p_ls]
                frame_608p_ls = [cv2.resize(img1080, (608,342)) for img1080 in frame_1080p_ls]
                
                frame_1080p_ls = [frame_1080p_ls[3],frame_1080p_ls[4]]
                frame_608p_ls = [frame_608p_ls[3],frame_608p_ls[4]]
                
                timestamp = frame_stamps[3]
                
                # Analyze the set of frames
                if rotate_to_find_tc:
                    tmp=10
                else:
                    frame_bbox_ls = [flash_tv.run_detector(img[:,:,::-1]) for img in frame_1080p_ls]
                
                if any(frame_bbox_ls):
                    face_seen_last_time = datetime.now()
                    frame_bbox_ls =  [flash_tv.run_verification(img[:,:,::-1], bbox_ls) for img, bbox_ls in zip(frame_1080p_ls, frame_bbox_ls)] 
                    
                    # perform gaze estimation if the child is there
                    tc_present, gz_data, tc_bbox, tc_id = flash_tv.run_gaze(frame_1080p_ls, frame_bbox_ls)
                    num_faces = 0 if tc_id<0 else len(frame_bbox_ls[tc_id])
                    
                    if tc_present:
                        label = 'Gaze-det'
                        #write the gaze and bbx logs
                        o1, e1, o2, e2 = gz_data
                        gaze_data1 = list(o1[0]) + [e1[0][0]]
                        gaze_data2 = list(o2[0]) + [e2[0][0]]
                        tc_angle = tc_bbox['angle']
                        
                        if abs(tc_angle)>=30:
                            #print('face rotation tranform!')
                            gaze_data1 = correct_rotation(gaze_data1, tc_angle)
                            gaze_data2 = correct_rotation(gaze_data2, tc_angle)
                            
                        tc_pos = [tc_bbox['top'], tc_bbox['left'], tc_bbox['bottom'], tc_bbox['right']]
                        print('%s, %06d, %.3f, %.3f, %s'%(str(timestamp), frame_counts[3], gaze_data1[0], gaze_data1[1], label))
                    else:
                        #write the gaze no det logs
                        label = 'Gaze-no-det'
                        gaze_data1 = [None]*3; gaze_data2 = [None]*3;
                        tc_pos = [None]*4; tc_angle = None
                        
                        print('%s, %06d, %s, %s, %s'%(str(timestamp), frame_counts[3], gaze_data1[0], gaze_data1[1], label))
                    
                    write_line = [timestamp, str(frame_counts[3]).zfill(6), num_faces, int(tc_present)] + gaze_data1 + [tc_angle] + tc_pos + [label]
                    
                else:
                    label = 'No-face-detected'
                    num_faces = 0; tc_present = 0;
                    print('%s, %06d, %s'%(str(timestamp), frame_counts[3], label))
                    write_line = [timestamp, str(frame_counts[3]).zfill(6), num_faces, tc_present, None, None, None, None, None, None, None, None, label]
                
                if write_image_data:
                    save_path = os.path.join(frames_save_path, str(frame_counts[3]).zfill(6) + '.png')
                    if tc_present:
                        _, _ = draw_gz(frame_1080p_ls[tc_id], np.array(gaze_data1).reshape(1,3), tc_bbox, save_path, gz_label=None, write_img=True, scale=[480, 854])
                    else:
                        _ = draw_rect_ver(frame_1080p_ls[1], frame_bbox_ls[1], None, save_path, write_img=True, scale=[480, 854])
                        #if num_faces>0:
                        #else:
                    
                log_lines.append(write_line)
                if len(log_lines)==5:
                    write_log_file(log_path, log_lines)
                    log_lines = []
            else:
                if qempty_start is None:
                        qempty_start = time.time()
                else:
                    empty_duration = time.time()-qempty_start
                    if empty_duration > 45:
                        print('*************************************************************************************')
                        print('CAMERA QUEUE DID NOT LOAD; RESTARTING')
                        print('*************************************************************************************')
                        stop_capture = True
                        time.sleep(5)
                        p1.join()
                        del q
                        break
    except Exception:
        traceback.print_exc()
        log_file[1] = frame_counter
        stop_capture = True
        time.sleep(5)
        p1.join()
        del q
        break     
