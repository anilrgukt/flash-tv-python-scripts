import sys 
import torch
import numpy as np 

from PIL import Image

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.utils as vutils

sys.path.insert(1, './gaze/')
from gaze.model import GazeLSTM, GazeLSTMreg

checkpoint_r50 = '/home/flashsys1/gaze_models/model_v3_best_Gaze360ETHXrtGene_r50.pth.tar'
checkpoint_r50reg = '/home/flashsys1/gaze_models/model_v3_best_Gaze360ETHXrtGene_r50reg.pth.tar'
cudnn.benchmark = True

model_v = GazeLSTM()
model = torch.nn.DataParallel(model_v).cuda()
        
checkpoint = torch.load(checkpoint_r50)
print('epochs', checkpoint['epoch'])
model.load_state_dict(checkpoint['state_dict'])
model.eval()
        
modelregv = GazeLSTMreg()
modelreg = torch.nn.DataParallel(modelregv).cuda()
checkpoint = torch.load(checkpoint_r50reg)
print('epochs', checkpoint['epoch'])
modelreg.load_state_dict(checkpoint['state_dict'])
modelreg.eval()

gaze_models = [model, modelreg]
        



class FLASHGazeEstimator():
    def __init__(self, num_gaze_models=2, img_size=224, vid_res=7):
        #self.num_gaze_models = num_gaze_models      
        self.gaze_models = gaze_models[0:num_gaze_models]  
        self.img_size = img_size
        self.vid_res = vid_res
        
        self.image_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.image_transform = transforms.Compose([transforms.Resize((self.img_size, self.img_size)),transforms.ToTensor(),self.image_normalize,])
        
    def to_input(self, tc_images):
    
        
        source_video_7fps = torch.FloatTensor(self.vid_res,3,self.img_size,self.img_size) 
        
        tc_images = tc_images*7
        for idx, im in enumerate(tc_images):        
            im = Image.fromarray(im)
            source_video_7fps[idx,...] = self.image_transform(im)
        
        source_video_7fps = source_video_7fps.view(self.vid_res*3,self.img_size,self.img_size)
        return source_video_7fps
        
    def gaze_estimate(self, source_video):
        source_video = torch.unsqueeze(source_video, 0)
        source_frame = source_video.cuda(non_blocking=True)
        
        with torch.no_grad():
            source_frame_var = torch.autograd.Variable(source_frame)
            #output = [m(source_frame_var) for m in self.gaze_models]
            output, ang_error = self.gaze_models[0](source_frame_var)
            output_, ang_error_ = self.gaze_models[1](source_frame_var)

        return [(output, ang_error), (output_, ang_error_)]





