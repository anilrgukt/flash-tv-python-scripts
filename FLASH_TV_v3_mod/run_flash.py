import os
import pickle 
import time

import cv2
import numpy as np 

from face_detection import FlashFaceDetector
from face_verification import FLASHFaceVerification
from gaze_estimation import FLASHGazeEstimator, load_limits, get_lims, eval_thrshld
from face_processing import FaceModelv4 as FaceProcessing
from utils.bbox_utils import Bbox
from utils.visualizer import draw_rect_det, draw_rect_ver, draw_gz

from utils.stream import WebcamVideoStream



from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

'''
data_path = '/media/flashsys007/FLASH_SSD/112_data'
frame_path = '/media/flashsys007/FLASH_SSD/112_data/112_frames'
det_res_path = '/media/flashsys007/FLASH_SSD/112_data/112_detres'
det_bbx_path = '/media/flashsys007/FLASH_SSD/112_data/112_detres_bbx'
fv_res_path = '/media/flashsys007/FLASH_SSD/112_data/112_fvres'
gz_res_path = '/media/flashsys007/FLASH_SSD/112_data/112_gzres'
'''

data_path = '/home/flashsys007/dmdm2023/data'
frame_path = '/home/flashsys007/dmdm2023/data/frames'
det_res_path = '/home/flashsys007/dmdm2023/data/detres'
det_bbx_path = '/home/flashsys007/dmdm2023/data/detres_bbx'
fv_res_path = '/home/flashsys007/dmdm2023/data/fvres'
gz_res_path = '/home/flashsys007/dmdm2023/data/gzres'


vis = True
write_image = True

cam_stream = False
save_bbx = not cam_stream
skip_detector = True

frame_ls = os.listdir(frame_path)
frame_ls.sort()

fd = FlashFaceDetector() #detector_hw=[480,860]) #detector_hw=[720,1280]) #detector_hw=[480,860])
fv = FLASHFaceVerification() #verification_threshold=0.516)
gz = FLASHGazeEstimator()

if not skip_detector:
    faces, lmarks = fd.face_detect(cv2.imread('frame_00000.png'))


face_processing = FaceProcessing(frame_resolution=[1080,1920], detector_resolution=[342,608],
                                 face_size=112, face_crop_offset=16, small_face_padding=7, small_face_size=65)

gaze_face_processing = FaceProcessing(frame_resolution=[1080,1920], detector_resolution=[342,608], 
                                    face_size=160, face_crop_offset=45, small_face_padding=3, small_face_size=65)                                 


loc_lims = load_limits(file_path='./4331_v3r50reg_reg_testlims_35_53_7_9.npy', setting='center-big-med')
num_locs = loc_lims.shape[0]

gt_embedding = fv.get_gt_emb(fam_id='123',path=data_path,face_proc=face_processing)

stream = WebcamVideoStream()
stream.start(src='/dev/video0', width=1920, height=1080, fps=30)

if not cam_stream:
    time.sleep(3)
    stream.stop()

plt.ylim([-.25,1.25])
#
#frame_id = 0
#while True:
for frame_id, frame_name in enumerate(frame_ls):    
    if cam_stream:
        frame_name = 'sframe_%05d.png'%(frame_id)
        img_cv1080 = stream.read()
        frame_time = datetime.now()#.strftime("%Y-%m-%d %H:%M:%S")
    else:
        img_path = os.path.join(frame_path, frame_name)
        img_cv1080 = cv2.imread(img_path)
        frame_time = datetime.now()
    
    img_cv608 = cv2.resize(img_cv1080, (608,342))
    img_np608 = img_cv608[:,:,::-1]    
    
    
    if skip_detector:
        fi = open(os.path.join(det_bbx_path, frame_name+'.pickle'),'rb')
        bls = pickle.load(fi)
    else:
        faces, lmarks = fd.face_detect(img_cv1080)
        bls = fd.convert_bbox(faces, lmarks)
        
        fi = open(os.path.join(det_bbx_path, frame_name+'.pickle'),'wb')
        pickle.dump(bls, fi)
        
    
    bbox_ls = [Bbox(b) for b in bls]
    
    cropped_aligned_faces = []
    check_faces = []
    for bbx in bbox_ls:
        #draw_rect_det(img_np608, [bbx.return_dict()], os.path.join(det_res_path, frame_name[:-4]+'_c608.png'))
        
        face, bbx_ = face_processing.crop_face_from_frame(img_cv1080, bbx)
        #draw_rect_det(img_cv1080[:,:,::-1], [bbx_.return_dict()], os.path.join(det_res_path, frame_name[:-4]+'_c1080.png'))
        
        
        check_faces.append(face)
        face, lmarks = face_processing.resize_face(face, bbx_)
        facen = face_processing.get_normalized_face(face, lmarks.astype(np.int).reshape(1,5,2), face=True)
        
        cropped_aligned_faces.append(facen)
        cropped_aligned_faces.append(facen[:,::-1,:])
    
    #for idx, face_ in enumerate(check_faces):
    #    cv2.imwrite('./tmp/%s_%d.png'%(frame_name[:-4], idx), face_)
        
    cropped_aligned_faces = np.array(cropped_aligned_faces)
    #print(cropped_aligned_faces.shape)
    #fv_inputs = [fv.to_input(face) for face in cropped_aligned_faces]
    
    det_emb, cropped_aligned_faces = fv.get_face_embeddings(cropped_aligned_faces)
    #print(cropped_aligned_faces.shape)
    #print(det_emb.shape, gt_embedding.shape)
    
    
    pred_ids, _, _ = fv.convert_embedding_faceid(ref_features=gt_embedding, test_features=det_emb, gal_update=[False,False,False], mean=0)
    
    for i in range(len(bls)):
        bls[i]['idx'] = pred_ids[i]
        
    tc_face = None
    tc_bbx = None
    for bbx in bls:
        if bbx['idx'] == 0:
            bbx_ = Bbox(bbx)
            face, bbx_ = gaze_face_processing.crop_face_from_frame(img_cv1080, bbx_)
            face, lmarks = gaze_face_processing.resize_face(face, bbx_)
            face_rot, angle = gaze_face_processing.rotate_face(face, lmarks, angle=None)
            bbx['angle'] = angle
            
            #tc_faces.append(face_rot)
            tc_face = face_rot
            tc_bbx = bbx
    
    #print(frame_id, frame_name, pred_ids, tc_bbx==None)
    
    
    #draw_rect_ver(img_np608, bls, None, os.path.join(fv_res_path, frame_name))
    if tc_face is not None:
        cv2.imwrite('./tmp/%s.png'%(frame_name[:-4]), tc_face)
        gaze_input = gz.to_input([tc_face])
        output = gz.gaze_estimate(gaze_input)
        o1, e1 = output[0]
        o2, e2 = output[1]
        o1 = o1.cpu().data.numpy()
        e1 = e1.cpu().data.numpy()
        #print(o1)
        #print(o.shape, e.shape)
        
        lims_idx = get_lims(tc_bbx, num_locs, H=342, W=608)
        _, _, gaze_est, _, _ = eval_thrshld(np.array([o1[0,0]]), np.array([o1[0,1]]), gt_lab=np.array([0]), lims=loc_lims[lims_idx])
        
        
        result_img = draw_gz(img_np608, o1, tc_bbx, os.path.join(gz_res_path, frame_name), gz_label=gaze_est[0], write_img=write_image)
        #print(frame_id, o1[0,0], o1[0,1], 'Gaze-detected', gaze_est[0])
        print('%s, Gaze-label: %d'%(frame_time.strftime("%Y-%m-%d %H:%M:%S"), gaze_est[0]))
        
        if gaze_est[0]:
            plt.plot(frame_time.strftime("%Y-%m-%d %H:%M:%S.f"), 1, 'o',color='g')
            plt.pause(0.05)
        else:   
            plt.plot(frame_time.strftime("%Y-%m-%d %H:%M:%S.f"), 1, 'd',color='b')
            plt.pause(0.05)
    else:
        tmp=10
        #draw_rect_det(img_np608, bls, os.path.join(det_res_path, frame_name))
        result_img = draw_rect_ver(img_np608, bls, None, os.path.join(gz_res_path, frame_name))
        #print(frame_id, None, None, 'Child-not-detected')
        print('%s, Gaze-label: %s'%(frame_time.strftime("%Y-%m-%d %H:%M:%S"), 'child-not-detected'))
    
    if vis:
        cv2.imshow('Gaze Result', result_img)
        pressedKey = cv2.waitKey(1) & 0xFF
        if pressedKey == ord('q'):
            break
    
    if cam_stream:
        frame_id = frame_id+1
    
    #im1.set_data(result_img[:,:,::-1])
    #plt.pause(0.01)
plt.show()
stream.stop()

#print(len(frame_ls))
#plt.ioff() # due to infinite loop, this gets never called.
#plt.show()


'''
img = cv2.imread('/home/flashsys1/Desktop/FLASH_TV_v3_mod/000015.png')
img2 = cv2.resize(img, (608,342))

faces, lmarks = fd.face_detect(img)
bls = fd.convert_bbox(faces, lmarks)

draw_rect_det(img2[:,:,::-1], bls, 'tmp_det.png')
'''
