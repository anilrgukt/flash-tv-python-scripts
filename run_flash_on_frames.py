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
from utils.rotate_frame import rotate_frame

from utils.flash_runtime_utils import check_face_presence, cam_id, write_log_file, correct_rotation, make_directories
from utils.visualizer import draw_rect_ver, draw_gz
        
# super variables 
write_image_data = True
rotate_to_find_tc = True
famid = 123

famid = str(famid)
fname_log = famid + '_flash_logv9n.txt'
fname_log_read = famid + '_flash_log_sub_sort.txt'
a = open(fname_log_read,'r')
q = a.readlines()
q = [line.strip() for line in q]
q = q[::-1]


save_path = '/home/'+os.getlogin()+'/dmdm2023/data'
frames_path = os.path.join(save_path, str(famid)+'_frames')
frames_save_path = os.path.join(save_path, str(famid)+'_test_res') # '/media/FLASH_SSD/525_test_res/'
frames_path, frames_save_path = make_directories(save_path, famid, frames_path, frames_save_path)


tmp_fname = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
tmp_fname = '_'.join(tmp_fname.split(' '))

log_path = os.path.join(save_path, str(famid) + '_flash_log_'+tmp_fname+'.txt')
log_path_reg = os.path.join(save_path, str(famid) + '_flash_log_'+tmp_fname+'_reg.txt')
log_path_rot = os.path.join(save_path, str(famid) + '_flash_log_'+tmp_fname+'_rot.txt') 

num_identities = 4
flash_tv = FLASHtv(family_id=str(famid), num_identities=num_identities, data_path=save_path, frame_res_hw=None, output_res_hw=None)
rot_frame = rotate_frame()

frame_counter = 1
log_file = [log_path, frame_counter]

frame_counter = log_file[1] 
log_path = log_file[0]
face_seen_last_time = datetime.now()
        
log_lines = []
log_lines_rot = []
log_lines_reg = []
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
        log_file[1] = frame_counter
        stop_capture = True
        time.sleep(5)
        p1.join()
        del q
        break
    
    for idx in range(num_identities):
        time_diff = datetime.now() - flash_tv.fv.gal_updated_time[idx]
        if time_diff.total_seconds() >= 150.0:
            flash_tv.fv.gal_update[idx] = True
    
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
        
        frame_1080p_ls = [frame_1080p_ls[3],frame_1080p_ls[4]] # analyze only two images cause of real-time constraints
        frame_608p_ls = [frame_608p_ls[3],frame_608p_ls[4]]
        
        timestamp = frame_stamps[3]
        
        # Analyze the set of frames
        if rotate_to_find_tc:
            img1, img2 = frame_1080p_ls
            frame_bbox_ls = [flash_tv.run_detector(img1[:,:,::-1])]
            
            img2 = rot_frame.rotate(img2)
            bbox2 = flash_tv.run_detector(img2[:,:,::-1], now_threshold=0.11)
            bbox2 = rot_frame.rotate_transform(bbox2) if rot_frame.rotate_flip>=0 else bbox2
            frame_bbox_ls.append(bbox2)
        else:
            frame_bbox_ls = [flash_tv.run_detector(img[:,:,::-1]) for img in frame_1080p_ls]
            rot_frame.rotate_count = 0
            rot_frame.rotate_count_tc = 0
            rot_frame.rotate_flip = False
        
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
                
                #print('face rotation tranform!')
                gaze_data1_rot = correct_rotation(gaze_data1, tc_angle) if abs(tc_angle)>=30 else gaze_data1
                gaze_data2_rot = gaze_data2 #correct_rotation(gaze_data2, tc_angle) if abs(tc_angle)>=30 else gaze_data2
                    
                tc_pos = [tc_bbox['top'], tc_bbox['left'], tc_bbox['bottom'], tc_bbox['right']]
                print('%s, %06d, %.3f, %.3f, %s'%(str(timestamp), frame_counts[3], gaze_data1[0], gaze_data1[1], label))
            else:
                #write the gaze no det logs
                label = 'Gaze-no-det'
                gaze_data1 = [None]*3; gaze_data2 = [None]*3;
                gaze_data1_rot = [None]*3; gaze_data2_rot = [None]*3;
                tc_pos = [None]*4; tc_angle = None
                
                print('%s, %06d, %s, %s, %s'%(str(timestamp), frame_counts[3], gaze_data1[0], gaze_data1[1], label))
            
            write_line = [timestamp, str(frame_counts[3]).zfill(6), num_faces, int(tc_present)] + gaze_data1 + [tc_angle] + tc_pos + [label]
            write_line_rot = [timestamp, str(frame_counts[3]).zfill(6), num_faces, int(tc_present)] + gaze_data1_rot + [tc_angle] + tc_pos + [label]
            write_line_reg = [timestamp, str(frame_counts[3]).zfill(6), num_faces, int(tc_present)] + gaze_data2 + [tc_angle] + tc_pos + [label]
            
            log_lines_rot.append(write_line_rot)
            log_lines_reg.append(write_line_reg)
            
        else:
            label = 'No-face-detected'
            num_faces = 0; tc_present = 0; tc_id = -1;
            print('%s, %06d, %s'%(str(timestamp), frame_counts[3], label))
            write_line = [timestamp, str(frame_counts[3]).zfill(6), num_faces, tc_present, None, None, None, None, None, None, None, None, label]
        
        if rotate_to_find_tc:
            num_faces_frame2 = len(frame_bbox_ls[1])
            tc_present_frame2 = tc_id == 2
            rot_frame.update(tc_present_frame2, num_faces_frame2)
        
        if write_image_data:
            save_path = os.path.join(frames_save_path, str(frame_counts[3]).zfill(6) + '.png')
            if tc_present:
                _, _ = draw_gz(frame_1080p_ls[tc_id], np.array(gaze_data1).reshape(1,3), tc_bbox, save_path, gz_label=None, write_img=True, scale=[480, 854])
            else:
                _ = draw_rect_ver(frame_1080p_ls[1], frame_bbox_ls[1], None, save_path, write_img=True, scale=[480, 854])
            
        log_lines.append(write_line)
        if len(log_lines)==5:
            write_log_file(log_path, log_lines)
            write_log_file(log_path_rot, log_lines_rot)
            write_log_file(log_path_reg, log_lines_reg)
            
            log_lines = []
            log_lines_rot = []
            log_lines_reg = []
