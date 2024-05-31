from face_detection import FlashFaceDetector
import cv2
import numpy as np 
from utils.visualizer import draw_rect_det
from utils.stream import WebcamVideoStream

import time

fd = FlashFaceDetector(detector_hw=[480,860])

'''
img = cv2.imread('/home/flashsys1/Desktop/FLASH_TV_v3_mod/000015.png')
img2 = cv2.resize(img, (608,342))

faces, lmarks = fd.face_detect(img)
bls = fd.convert_bbox(faces, lmarks)
print(bls)


draw_rect_det(img2[:,:,::-1], bls, 'tmp_det.png', draw_lmarks=True)
'''



img = cv2.imread('./frame_00000.png') #video_reader.read()
faces, lmarks = fd.face_detect(img)

idx = 0
video_reader = WebcamVideoStream()
video_reader.start(idx, width=1920, height=1080, fps=5)

t = time.time()
for i in range(100):
    img = video_reader.read()
    imgr = cv2.resize(img, (608,342))
    print(i, img.shape)
    #img_cap_time = datetime.now()

    faces, lmarks = fd.face_detect(img)
    bls = fd.convert_bbox(faces, lmarks)
    #print(bls)
    draw_rect_det(imgr[:,:,::-1], bls, 'det_%05d.png'%i, draw_lmarks=True)

video_reader.stop()    
print((time.time()-t)/100)
