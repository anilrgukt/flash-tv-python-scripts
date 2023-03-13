import os
import pickle 
import time

import cv2
import numpy as np 

from face_detection import FlashFaceDetector
from face_verification import FLASHFaceVerification
from gaze_estimation import FLASHGazeEstimator
from face_processing import FaceModelv4 as FaceProcessing
from utils.bbox_utils import Bbox
from utils.visualizer import draw_rect_det, draw_rect_ver, draw_gz

data_path = '/media/FLASH_SSD/112_data'
frame_path = '/media/FLASH_SSD/112_data/112_frames'
det_res_path = '/media/FLASH_SSD/112_data/112_detres'
det_bbx_path = '/media/FLASH_SSD/112_data/112_detres_bbx'
fv_res_path = '/media/FLASH_SSD/112_data/112_fvres'
gz_res_path = '/media/FLASH_SSD/112_data/112_gzres'




class FLASHtv():
    def __init__(self, family_id, frame_res_hw, output_res_hw):
        self.fd = FlashFaceDetector()
        self.fv = FLASHFaceVerification()
        self.gz = FLASHGazeEstimator()
        
        self.face_processing = FaceProcessing(frame_resolution=[1080,1920], detector_resolution=[342,608], face_size=112,
                                face_crop_offset=16, small_face_padding=7, small_face_size=65)
        self.gaze_face_processing = FaceProcessing(frame_resolution=[1080,1920], detector_resolution=[342,608], face_size=160,
                                face_crop_offset=45, small_face_padding=3, small_face_size=65)          
        
        self.family_id = family_id
        self.data_path = data_path
        
        # get the GT embedding ... 
        self.gt_embedding = self.fv.get_gt_emb(fam_id=self.family_id, path=self.data_path, face_proc=self.face_processing)
        
    def run_detector(self, []):
        
        
        
        faces, lmarks = self.fd.face_detect(img_cv1080)
        frame_bls = self.fd.convert_bbox(faces, lmarks)
        
        return frame_bls
        
    def run_verification(self, []):
    
        bbox_ls = [Bbox(b) for b in bls]
        
        cropped_aligned_faces = []
        for bbx in bbox_ls:
            face, bbx_ = face_processing.crop_face_from_frame(img_cv1080, bbx)
            face, lmarks = face_processing.resize_face(face, bbx_)
            facen = face_processing.get_normalized_face(face, lmarks.astype(np.int).reshape(1,5,2), face=True)
            
            cropped_aligned_faces.append(facen)
            cropped_aligned_faces.append(facen[:,::-1,:])
            
            
        cropped_aligned_faces = np.array(cropped_aligned_faces)
        print(cropped_aligned_faces.shape)
        #fv_inputs = [fv.to_input(face) for face in cropped_aligned_faces]
        
        det_emb, cropped_aligned_faces = fv.get_face_embeddings(cropped_aligned_faces)
        print(cropped_aligned_faces.shape)
        print(det_emb.shape, gt_embedding.shape)
        
        
        pred_ids, _, _ = fv.convert_embedding_faceid(ref_features=gt_embedding, test_features=det_emb, gal_update=[False,False,False], mean=0)
        
        print(pred_ids)
        for i in range(len(bls)):
            bls[i]['idx'] = pred_ids[i]
            
        return bls, cropped_aligned_faces
        
    def run_tcface(self, []):
        
            
            










frame_ls = os.listdir(frame_path)






fd = FlashFaceDetector()
fv = FLASHFaceVerification()
gz = FLASHGazeEstimator()

face_processing = FaceProcessing(frame_resolution=[1080,1920], detector_resolution=[342,608], face_size=112,
                                 face_crop_offset=16, small_face_padding=7, small_face_size=65)

gaze_face_processing = FaceProcessing(frame_resolution=[1080,1920], detector_resolution=[342,608], face_size=160,
                                 face_crop_offset=45, small_face_padding=3, small_face_size=65)                                 

gt_embedding = fv.get_gt_emb(fam_id='112',path=data_path,face_proc=face_processing)

for i, frame_name in enumerate(frame_ls):
    print(i)
    
    img_path = os.path.join(frame_path, frame_name)
    
    img_cv1080 = cv2.imread(img_path)
    
    t = time.time()
    img_cv608 = cv2.resize(img_cv1080, (608,342))
    
    faces, lmarks = fd.face_detect(img_cv1080)
    bls = fd.convert_bbox(faces, lmarks)
    
    #fi = open(os.path.join(det_bbx_path, frame_name+'.pickle'),'wb')
    #pickle.dump(bls, fi)
    
    #fi = open(os.path.join(det_bbx_path, frame_name+'.pickle'),'rb')
    #bls = pickle.load(fi)
    
    bbox_ls = [Bbox(b) for b in bls]
    
    cropped_aligned_faces = []
    for bbx in bbox_ls:
        face, bbx_ = face_processing.crop_face_from_frame(img_cv1080, bbx)
        face, lmarks = face_processing.resize_face(face, bbx_)
        facen = face_processing.get_normalized_face(face, lmarks.astype(np.int).reshape(1,5,2), face=True)
        
        cropped_aligned_faces.append(facen)
        cropped_aligned_faces.append(facen[:,::-1,:])
        
        
    cropped_aligned_faces = np.array(cropped_aligned_faces)
    print(cropped_aligned_faces.shape)
    #fv_inputs = [fv.to_input(face) for face in cropped_aligned_faces]
    
    det_emb, cropped_aligned_faces = fv.get_face_embeddings(cropped_aligned_faces)
    print(cropped_aligned_faces.shape)
    print(det_emb.shape, gt_embedding.shape)
    
    
    pred_ids, _, _ = fv.convert_embedding_faceid(ref_features=gt_embedding, test_features=det_emb, gal_update=[False,False,False], mean=0)
    
    print(pred_ids)
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

    img_np608 = img_cv608[:,:,::-1]    
    if tc_face is not None:
        gaze_input = gz.to_input([tc_face])
        output = gz.gaze_estimate(gaze_input)
        o, e = output[0]
        o = o.cpu().data.numpy()
        e = e.cpu().data.numpy()
        #print(o.shape, e.shape)
        draw_gz(img_np608, o, tc_bbx, os.path.join(gz_res_path, frame_name))
    else:
        #draw_rect_det(img_np608, bls, os.path.join(det_res_path, frame_name))
        draw_rect_ver(img_np608, bls, None, os.path.join(gz_res_path, frame_name))
        
    print(time.time()-t, 'secs')

    
    



'''
img = cv2.imread('/home/flashsys1/Desktop/FLASH_TV_v3_mod/000015.png')
img2 = cv2.resize(img, (608,342))

faces, lmarks = fd.face_detect(img)
bls = fd.convert_bbox(faces, lmarks)

draw_rect_det(img2[:,:,::-1], bls, 'tmp_det.png')
'''
