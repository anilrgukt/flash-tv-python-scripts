import cv2
import numpy as np

import math

from skimage.transform import resize
from imageio import imread, imsave

def draw_rect_det(img, dboxes, save_file, draw_lmarks=False):
    cv_img = np.copy(img)
    tmp_channel = np.copy(cv_img[:,:,0])
    cv_img[:,:,0] = cv_img[:,:,2]
    cv_img[:,:,2] = tmp_channel  
    colors = [(0,0,255),(0,255,0),(255,0,0),(0,0,0),(255,255,255)]
    for i, dbox in enumerate(dboxes):
        if dbox['prob']>0.03:
            cv2.rectangle(cv_img, (int(dbox["left"]), int(dbox["top"])), (int(dbox["right"]), int(dbox["bottom"])), (0,0,255), 2) 
        #cv2.circle(cv_img, (int(0.5 * (int(dbox["left"]) + int(dbox["right"]))), int(0.5 * (int(dbox["top"]) + int(dbox["bottom"])))), 2, (0,0,255))
        lmarks = dbox['lmarks']
        if draw_lmarks:
            for li in range(lmarks.shape[0]):
                cv2.circle(cv_img, (lmarks[li][0],lmarks[li][1]), 1, colors[li], 2)

    cv2.imwrite(save_file, cv_img)
    return cv_img
    
def draw_rect_ver(img, dboxes1, dboxes2, save_file):
    cv_img = np.copy(img)
    tmp_channel = np.copy(cv_img[:,:,0])
    cv_img[:,:,0] = cv_img[:,:,2]
    cv_img[:,:,2] = tmp_channel
    
    l = [(0,255,0),(255,0,0),(255,255,0),(255,0,255),(0,0,255)]

    draw_lmarks = True
    for i, dbox in enumerate(dboxes1):
        cv2.rectangle(cv_img, (int(dbox["left"]), int(dbox["top"])), (int(dbox["right"]), int(dbox["bottom"])), l[dbox['idx']], 2) 
        lmarks = dbox['lmarks']
        lmcolor = [(0,0,255),(0,255,0),(255,0,0),(255,255,0),(0,0,0)] #rgbc,black
        if draw_lmarks:
            for lm in range(lmarks.shape[0]):  
                color = (0, 0, 255)
                if lm == 0 or lm == 3:
                    color = (0, 255, 0)
                cv2.circle(cv_img, (lmarks[lm,0], lmarks[lm,1]), 1, color, 2)

    if dboxes2 is not None:
        for i, dbox in enumerate(dboxes2):
   	        cv2.rectangle(cv_img, (int(dbox["left"])+2, int(dbox["top"])+2), (int(dbox["right"])+2, int(dbox["bottom"])+2), (0,255,0), 1) 

    cv2.imwrite(save_file, cv_img)
    return cv_img

def draw_gz(frm, gaze_angle, bbx, save_path):

    s0 = gaze_angle[0,0]
    s1 = gaze_angle[0,1]
    
    sx = (bbx['left'] + bbx['right'])/2 # 
    sy = (bbx['top'] + bbx['bottom'])/2
    
    x = math.cos(s1)*math.sin(s0) # -40*
    y = math.sin(s1) # -40*
    z = -math.cos(s1)*math.cos(s0)    
    
    start, end = (int(sx),int(sy)), (int(sx+x),int(sy+y))
    
    cv_img = frm[:,:,::-1]
    cv2.arrowedLine(cv_img, start, end, (0,0,255), 3, tipLength=0.5) 
    cv2.imwrite(save_path, cv_img)
    
    return cv_img
    
#draw_rect_gz(batch10_img608[i], (int(sx),int(sy)), (int(sx+x),int(sy+y)), color, os.path.join(BASE_PATH_IMG,output_name))
def draw_rect_gz(img, frm, start, end, color, save_file):
    cv_img = np.copy(img)
    tmp_channel = np.copy(cv_img[:,:,0])
    cv_img[:,:,0] = cv_img[:,:,2]
    cv_img[:,:,2] = tmp_channel

    #l = [(0,255,0),(255,0,0),(255,255,0),(255,0,255),(0,0,255)]
    l = [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(0,0,255)]

    if start is not None:
        cv2.arrowedLine(cv_img, start, end, l[color], 3, tipLength=0.5) 
    
    cv2.putText(cv_img,str(frm), (455,30), cv2.FONT_HERSHEY_PLAIN, 2, 255)
    cv2.imwrite(save_file, cv_img)
    return cv_img
'''    
def draw_rect_gz(img, frm, start, end, color, save_file):
    cv_img = np.copy(img)
    tmp_channel = np.copy(cv_img[:,:,0])
    cv_img[:,:,0] = cv_img[:,:,2]
    cv_img[:,:,2] = tmp_channel

    #l = [(0,255,0),(255,0,0),(255,255,0),(255,0,255),(0,0,255)]
    l = [(255,0,0),(0,255,0),(0,0,255),(255,0,255),(0,0,255)]

    if start is not None:
        cv2.arrowedLine(cv_img, start, end, l[color], 3, tipLength=0.5) 
    
    cv2.putText(cv_img,str(frm), (455,30), cv2.FONT_HERSHEY_PLAIN, 2, 255)
    cv2.imwrite(save_file, cv_img)
    return cv_img
'''
