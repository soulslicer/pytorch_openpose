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
exit = 0
def signal_handler(sig, frame):
    global exit
    exit = 1
signal.signal(signal.SIGINT, signal_handler)
sys.path.append('/home/raaj/openpose_orig/build/python/')
from openpose import pyopenpose as op

from models import *

# Parsers
parser = argparse.ArgumentParser(description='OP')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--reload', action='store_true')
args = parser.parse_args()

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
model = Model(Body25("7x7"), ngpu=int(args.ngpu)).cuda()
model.train()

# Load weights etc.
iterations = 0
reload = int(args.reload)
if not reload:
    state = load_checkpoint(NAME)
    if state != None:
        iterations = state["iterations"]
        model.load_state_dict(state['state_dict'])
        print("Loaded Iteration " + str(iterations))

# # Load Caffe?
# model.net.load_caffe()

params = {
    "batch_size" : 10,
    "stride": 8,
    "max_degree_rotations": "45.0",
    "crop_size_x": 368,
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

# Loss
lr = 0.000100
parameters = [
        {"params": model.net.vgg19.parameters(), "lr": lr*1},
        {"params": model.net.pafA.parameters(), "lr": lr*4},
        {"params": model.net.pafB.parameters(), "lr": lr*4},
        {"params": model.net.hmA.parameters(), "lr": lr*4},
        {"params": model.net.hmB.parameters(), "lr": lr*4},
    ]
mseLoss = torch.nn.MSELoss()
optimizer = optim.Adam(parameters, lr=lr, betas=(0.9, 0.999))

# Iterate
while 1:
    iterations += 1
    batch = opcaffe.Batch()
    myClass.load(batch)

    # Split
    paf_mask = torch.tensor(batch.label[:, 0:TOTAL_PAFS]).cuda()
    hm_mask = torch.tensor(batch.label[:, TOTAL_PAFS:TOTAL_PAFS+TOTAL_HMS]).cuda()
    paf_truth = torch.tensor(batch.label[:, TOTAL_PAFS+TOTAL_HMS:TOTAL_PAFS+TOTAL_HMS+TOTAL_PAFS]).cuda()
    hm_truth = torch.tensor(batch.label[:, TOTAL_PAFS+TOTAL_HMS+TOTAL_PAFS:TOTAL_PAFS+TOTAL_HMS+TOTAL_PAFS+TOTAL_HMS]).cuda()
    imgs = torch.tensor(batch.data).cuda()

    # Mask
    paf_truth_m = torch.mul(paf_truth, paf_mask)
    hm_truth_m = torch.mul(hm_truth, hm_mask)

    # Forward Model
    pafs_pred, hms_pred = model.forward(imgs)

    # Multiply with Masks
    loss = 0
    for i in range(0, ITERATIONS):
        paf_pred_m = torch.mul(pafs_pred[i], paf_mask)
        hm_pred_m = torch.mul(hms_pred[i], hm_mask)
        loss += mseLoss(paf_pred_m, paf_truth_m)
        loss += mseLoss(hm_pred_m, hm_truth_m)

    # Opt
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save
    if iterations % 20 == 0 or exit:
        print("Saving")
        save_checkpoint({
            'iterations': iterations,
            'state_dict': model.state_dict(),
        }, NAME)
    if exit: sys.exit()
    print((iterations,loss))

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

    img_viz = imgs.detach().cpu().numpy().copy()[0,0,:,:]
    hm_pred_viz = hms_pred[ITERATIONS-1].detach().cpu().numpy().copy()[0,0,:,:]
    hm_truth_viz = hm_truth_m.cpu().numpy().copy()[0,0,:,:]
    cv2.imshow("hm_pred_viz", cv2.resize(hm_pred_viz, (0,0), fx=8, fy=8, interpolation = cv2.INTER_CUBIC))
    cv2.imshow("hm_truth_viz", cv2.resize(hm_truth_viz, (0,0), fx=8, fy=8, interpolation = cv2.INTER_CUBIC))
    cv2.imshow("img", img_viz+0.5)
    cv2.waitKey(15)


"""
Training of POF?
"""