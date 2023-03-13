from face_detection import FlashFaceDetector
import cv2
import numpy as np 
from utils.visualizer import draw_rect_ver

fd = FlashFaceDetector()

img = cv2.imread('/home/flashsys1/Desktop/FLASH_TV_v3_mod/000015.png')
img2 = cv2.resize(img, (608,342))

faces, lmarks = fd.face_detect(img)
bls = fd.convert_bbox(faces, lmarks)

draw_rect_ver(img2, bls, 'tmp_det.png')
