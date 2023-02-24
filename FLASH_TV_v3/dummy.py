import torch
import mxnet as mx
import sys
import os
#import blah

# face detection libs
sys.path.insert(1, '/home/flashsys002/insightface/detection/RetinaFace')
from retinaface import RetinaFace


from datetime import datetime
import time 

famid = sys.argv[1]
path = sys.argv[2]
file_location = os.path.join(path, str(famid)+'_check.txt')
c=0
while c<50: 
    l = datetime.now()
    print(l)
    time.sleep(5)
    
    fid = open(file_location,'a')
    fid.write(str(l)+'\n')
    fid.close()
    
    c=c+1
