from face_detection import FlashFaceDetector
import cv2
import numpy as np 
from utils.visualizer import draw_rect_det
from utils.stream import WebcamVideoStream

fd = FlashFaceDetector()

'''
img = cv2.imread('/home/flashsys1/Desktop/FLASH_TV_v3_mod/000015.png')
img2 = cv2.resize(img, (608,342))

faces, lmarks = fd.face_detect(img)
bls = fd.convert_bbox(faces, lmarks)
print(bls)


draw_rect_det(img2[:,:,::-1], bls, 'tmp_det.png', draw_lmarks=True)
'''


idx = 0
video_reader = WebcamVideoStream()
video_reader.start(idx, width=608, height=342)

img = video_reader.read()
img = cv2.resize(img, (608,342))
print(img.shape)
#img_cap_time = datetime.now()
video_reader.stop()

faces, lmarks = fd.face_detect(img)
bls = fd.convert_bbox(faces, lmarks)
print(bls)
draw_rect_det(img[:,:,::-1], bls, 'check_det.png', draw_lmarks=True)
