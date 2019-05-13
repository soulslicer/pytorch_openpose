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
from loader import *
import pof
import nms

# Parsers
parser = argparse.ArgumentParser(description='OP')
parser.add_argument('--weight', type=str, default="",
                    help='Weight')
args = parser.parse_args()

# Sample OP Network
params = dict()
params["model_folder"] = "/home/raaj/openpose_orig/models/"
params["body"] = 2  # Disable OP Network
params["upsampling_ratio"] = 0
params["model_pose"] = "BODY_25B"
params["number_people_max"] = 1
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Setup Model
NAME = "weights_pof"
model = Model(Gines(pof=True), ngpu=1).cuda()
model.eval()

# Load weights
if len(args.weight):
    state = torch.load(args.weight)
    if state != None:
        model.load_state_dict(state['state_dict'])
        print("Loaded State")
else:
    model.net.load_caffe()

# POF
pofBodyLoader = pof.POFBodyLoader(db_filename="human3d_test.pkl", batch_size=int(1), resolution=368)

import random
def decision(probability):
    return random.random() < probability

#capture from camera at location 0
cap = cv2.VideoCapture(0)
#set the width and height, and UNSUCCESSFULLY set the exposure time
# cap.set(3,368)
# cap.set(4,368)
# cap.set(15, 0.1)

while True:
    ret, img = cap.read()
    #cv2.imshow("thresholded", imgray*thresh2)

    # Process Image
    rframe, imageForNet, scaleFactor = process_frame(img, 368)
    imageForNet = torch.tensor(np.expand_dims(imageForNet, axis=0)).cuda()

    # Create Heatmaps
    pafA, pafB, pafC, hm, pofA, pofB = model.forward(imageForNet, True)  
    i=0   

    # Get Pose
    poseHeatMaps = torch.cat([hm[i,:,:,:], pafC[i,:,:,:]], 0).detach().cpu().numpy().copy()
    imageToProcess = imageForNet.detach().cpu().numpy().copy()[i,:,:,:]
    imageToProcess = (cv2.merge([imageToProcess[0,:,:]+0.5, imageToProcess[1,:,:]+0.5, imageToProcess[2,:,:]+0.5])*255).astype(np.uint8)
    datum = op.Datum()
    datum.name = "fag"
    datum.cvInputData = imageToProcess
    datum.poseNetOutput = poseHeatMaps
    opWrapper.emplaceAndPop([datum])

    print(len(datum.poseKeypoints.shape))

    if len(datum.poseKeypoints.shape) == 0: continue
    coord2d = datum.poseKeypoints[i,:,:].copy()


    coord3d = pof.PAF_to_3D(coord2d, pofB[i,:,:,:].detach().cpu().numpy(), 8)

    cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData)
    pof.viz_coord(coord3d, coord2d)
    cv2.waitKey(15)


    # print(imageForNet.shape)

    # cv2.imshow("input", rframe)


    # key = cv2.waitKey(10)
    # if key == 27:
    #     break

# Iterate
while 1:
    # Dataset Prob
    pof_mode = decision(1.0)

    # POF Mode
    if pof_mode:

        # Get Data
        images, paf_masks, pafs, pof_masks, pofs, hms_mask, hm = pofBodyLoader.get()

        # # Convert Torch
        imgs = torch.tensor(images).cuda()
        # paf_mask = torch.tensor(paf_masks).cuda()
        # pof_mask = torch.tensor(pof_masks).cuda()
        # paf_truth = torch.tensor(pafs).cuda()
        # pof_truth = torch.tensor(pofs).cuda()

        # Forward Model
        pafA, pafB, pafC, hm, pofA, pofB = model.forward(imgs, True)  
        i=0   

        # Image
        img = (cv2.merge([images[i,0,:,:]+0.5, images[i,1,:,:]+0.5, images[i,2,:,:]+0.5])*255).astype(np.uint8)
        cv2.imshow("img", img)

        # NMS
        pid = 0
        peaks = nms.NMS({'thre1': 0.05}, hm[i,:,:,:].detach().cpu().numpy(), 8)
        coord2d = np.zeros((25,3), dtype=np.float32)
        for j in range(0, 25):
            peak = peaks[j]
            if peak.shape[0] == 0: continue
            coord2d[j,:] = np.array([peak[pid][0], peak[pid][1], peak[pid][2]])

        coord3d = pof.PAF_to_3D(coord2d, pofB[i,:,:,:].detach().cpu().numpy(), 8, img)

        while 1:
            key = cv2.waitKey(15)
            pof.viz_coord(coord3d, coord2d)
            if key == 27:
                break
            #time.sleep(0.1)
        #cv2.waitKey(0)



        #pof.viz_hm(images[i,:,:,:], hm[i,:,:,:].detach().cpu().numpy())
        #pof.viz_pof(images[i,:,:,:], pofB[i,:,:,:].detach().cpu().numpy(), None)

        #cv2.waitKey(15)


        # # Opt
        # loss = 0
        # loss += mseLoss(torch.mul(pafA, paf_mask), paf_truth_m)
        # loss += mseLoss(torch.mul(pafB, paf_mask), paf_truth_m)
        # loss += mseLoss(torch.mul(pafC, paf_mask), paf_truth_m)
        # loss += mseLoss(torch.mul(pofA, pof_mask), pof_truth_m)  
        # loss += mseLoss(torch.mul(pofB, pof_mask), pof_truth_m)         

 
    # # OP Test
    # test_index = 0
    # hm_final = hms_pred[ITERATIONS-1][test_index,:,:,:]
    # paf_final = pafs_pred[ITERATIONS-1][test_index,:,:,:]
    # poseHeatMaps = torch.cat([hm_final, paf_final], 0).detach().cpu().numpy().copy()
    # imageToProcess = imgs.detach().cpu().numpy().copy()[test_index,:,:,:]
    # imageToProcess = (cv2.merge([imageToProcess[0,:,:]+0.5, imageToProcess[1,:,:]+0.5, imageToProcess[2,:,:]+0.5])*255).astype(np.uint8)
    # datum = op.Datum()
    # datum.cvInputData = imageToProcess
    # datum.poseNetOutput = poseHeatMaps
    # opWrapper.emplaceAndPop([datum])
    # print("Body keypoints: \n" + str(datum.poseKeypoints))
    # cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData)
    # cv2.waitKey(0)

    # img_viz = imgs.detach().cpu().numpy().copy()[0,0,:,:]
    # hm_pred_viz = hms_pred[ITERATIONS-1].detach().cpu().numpy().copy()[0,0,:,:]
    # hm_truth_viz = hm_truth_m.cpu().numpy().copy()[0,0,:,:]
    # cv2.imshow("hm_pred_viz", cv2.resize(hm_pred_viz, (0,0), fx=8, fy=8, interpolation = cv2.INTER_CUBIC))
    # cv2.imshow("hm_truth_viz", cv2.resize(hm_truth_viz, (0,0), fx=8, fy=8, interpolation = cv2.INTER_CUBIC))
    # cv2.imshow("img", img_viz+0.5)
    # cv2.waitKey(15)


"""
Training of POF?
"""