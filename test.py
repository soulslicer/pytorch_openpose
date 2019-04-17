from __future__ import print_function
import argparse
import os
import sys
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init 
import torchvision.models as models 
from torchvision import transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import scipy.io as sio
import time
from collections import OrderedDict
import numpy as np
import torch.utils.model_zoo as model_zoo
import os 
import cv2
os.environ['GLOG_minloglevel'] = '2' 
#import caffe
dir_path = os.path.dirname(os.path.realpath(__file__))
import sys
sys.path.insert(0, "/home/raaj/openpose_caffe_train/build/op/")
import opcaffe
import signal
sys.path.append('/home/raaj/openpose_orig/build/python/')
from openpose import pyopenpose as op

from models import *

# Sample OP Network
params = dict()
params["model_folder"] = "/home/raaj/openpose_orig/models/"
params["body"] = 2  # Disable OP Network
params["upsampling_ratio"] = 0
params["model_pose"] = "BODY_25B"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Setup Model
NAME = "weights"
model = Model(Body25("3x3"), ngpu=int(1)).cuda()
model.eval()
model.net.load_caffe()

params = {
    "batch_size" : 1,
    "stride": 8,
    "max_degree_rotations": "45.0",
    "crop_size_x": 656,
    "crop_size_y": 368,
    "center_perterb_max": 40.0,
    "center_swap_prob": 0.0,
    "scale_prob": 1.0,
    "scale_mins": "0.333333333333",
    "scale_maxs": "1.5",
    "target_dist": 0.600000023842,
    "number_max_occlusions": "2",
    "sigmas": "7.0",
    "models": "COCO_25B_23;COCO_25B_17;MPII_25B_16;PT_25B_15",
    "sources": "/media/raaj/Storage/openpose_train/dataset/lmdb_coco2017_foot;/media/raaj/Storage/openpose_train/dataset/lmdb_coco;/media/raaj/Storage/openpose_train/dataset/lmdb_mpii;/media/raaj/Storage/openpose_train/dataset/lmdb_pt2_train",
    "probabilities": "0.05;0.85;0.05;0.05",
    "source_background": "/media/raaj/Storage/openpose_train/dataset/lmdb_background",
    "normalization": 0,
    "add_distance": 0
}
myClass = opcaffe.OPCaffe(params)

# Iterate
while 1:
    batch = opcaffe.Batch()
    myClass.load(batch)
    imgs = torch.tensor(batch.data).cuda()

    torch.cuda.synchronize()
    start_time = time.time()
    pafs_pred, hms_pred = model.forward(imgs)
    torch.cuda.synchronize()
    print("--- %s seconds ---" % (time.time() - start_time))

    # OP Test
    test_index = 0
    hm_final = hms_pred[ITERATIONS-1][test_index,:,:,:]
    paf_final = pafs_pred[ITERATIONS-1][test_index,:,:,:]
    poseHeatMaps = torch.cat([hm_final, paf_final], 0).detach().cpu().numpy().copy()
    imageToProcess = imgs.detach().cpu().numpy().copy()[test_index,:,:,:]
    imageToProcess = (cv2.merge([imageToProcess[0,:,:]+0.5, imageToProcess[1,:,:]+0.5, imageToProcess[2,:,:]+0.5])*255).astype(np.uint8)
    datum = op.Datum()
    datum.cvInputData = imageToProcess
    datum.poseNetOutput = poseHeatMaps
    opWrapper.emplaceAndPop([datum])
    #print("Body keypoints: \n" + str(datum.poseKeypoints))
    cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData)
    cv2.waitKey(15)


    print("a")