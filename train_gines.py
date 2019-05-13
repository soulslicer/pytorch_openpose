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
import time
from torch.multiprocessing import Process, Queue, Value, cpu_count
os.environ['GLOG_minloglevel'] = '2' 
dir_path = os.path.dirname(os.path.realpath(__file__))

# Params
NAME = "weights_gines_no"
OP_CAFFE_TRAIN_PATH = '/home/raaj/openpose_caffe_train/build/op/'
OP_PYTHON_PATH = '/home/raaj/openpose_orig/build/python/'
OP_MODEL_FOLDER = '/home/raaj/openpose_orig/models/'
OP_LMDB_FOLDER = '/media/raaj/Storage/openpose_train/dataset/'
OP_RESOLUTION = 480

# Insert OP Paths
import sys
sys.path.insert(0, OP_CAFFE_TRAIN_PATH)
import opcaffe
import signal
exit = 0
def signal_handler(sig, frame):
    global exit
    exit = 1
signal.signal(signal.SIGINT, signal_handler)
sys.path.append(OP_PYTHON_PATH)
from openpose import pyopenpose as op

# Load Models
from models import *
from loader import *

# Parsers
parser = argparse.ArgumentParser(description='OP')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--batch', type=int, default=10,
                    help='batch size')
parser.add_argument('--debug', type=int, default=0,
                    help='debug')
parser.add_argument('--reload', action='store_true')
args = parser.parse_args()

# Sample OP Network
params = dict()
params["model_folder"] = OP_MODEL_FOLDER
params["body"] = 2  # Disable OP Network
params["upsampling_ratio"] = 0
params["model_pose"] = "BODY_25B"
params["net_resolution"] = "-1x"+str(OP_RESOLUTION)
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Setup Model
model = Model(Gines(), ngpu=int(args.ngpu)).cuda()
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

# Load Caffe?
# model.net.load_caffe()

params = {
    "batch_size" : int(args.batch),
    "stride": 8,
    "max_degree_rotations": "45.0",
    "crop_size_x": OP_RESOLUTION,
    "crop_size_y": OP_RESOLUTION,
    "center_perterb_max": 40.0,
    "center_swap_prob": 0.0,
    "scale_prob": 1.0,
    "scale_mins": "0.333333333333",
    "scale_maxs": "1.5",
    "target_dist": 0.600000023842,
    "number_max_occlusions": "2",
    "sigmas": "7.0",
    "models": "COCO_25B_23;COCO_25B_17;MPII_25B_16;PT_25B_15",
    "sources": OP_LMDB_FOLDER+"lmdb_coco2017_foot;"+OP_LMDB_FOLDER+"lmdb_coco;"+OP_LMDB_FOLDER+"lmdb_mpii;"+OP_LMDB_FOLDER+"lmdb_pt2_train",
    "probabilities": "0.05;0.85;0.05;0.05",
    "source_background": OP_LMDB_FOLDER+"lmdb_background",
    "normalization": 0,
    "add_distance": 0
}
myClass = opcaffe.OPCaffe(params)

# Loss
lr = 0.000020
parameters = [
        {"params": model.net.vgg19.parameters(), "lr": lr*1},
        {"params": model.net.pafA.parameters(), "lr": lr*4},
        {"params": model.net.pafB.parameters(), "lr": lr*4},
        {"params": model.net.pafC.parameters(), "lr": lr*4},
        {"params": model.net.hmNetwork.parameters(), "lr": lr*4},
    ]
mseLoss = torch.nn.MSELoss()
optimizer = optim.Adam(parameters, lr=lr, betas=(0.9, 0.999))
lr_half_sets = [200000, 300000, 360000, 420000, 480000, 540000, 600000, 700000, 800000]

def half_lr(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] /= 2.

# Data Worker
def work(loader, queue, control):
    while 1:
        if control.value == 0: 
            break
        if queue.qsize() < 5:
            batch = opcaffe.Batch()
            myClass.load(batch)
            data = torch.tensor(batch.data)
            label = torch.tensor(batch.label)
            queue.put([data, label])
        time.sleep(0.1)
queue = Queue()
control = Value('i',1)
process = Process(target=work, args=(myClass, queue, control))
process.start()

# Iterate
while 1:
    iterations += 1

    # Get Data from Queue
    data, label = queue.get()

    # LR
    if iterations in lr_half_sets:
        print("Half LR")
        half_lr(optimizer) 

    # Split
    bs = label.shape[0]
    paf_mask = label[0:bs, 0:TOTAL_PAFS].cuda()
    hm_mask = label[0:bs, TOTAL_PAFS:TOTAL_PAFS+TOTAL_HMS].cuda()
    paf_truth = label[0:bs, TOTAL_PAFS+TOTAL_HMS:TOTAL_PAFS+TOTAL_HMS+TOTAL_PAFS].cuda()
    hm_truth = label[0:bs, TOTAL_PAFS+TOTAL_HMS+TOTAL_PAFS:TOTAL_PAFS+TOTAL_HMS+TOTAL_PAFS+TOTAL_HMS].cuda()
    imgs = data[0:bs, :,:,:].cuda()

    # Mask
    paf_truth_m = torch.mul(paf_truth, paf_mask)
    hm_truth_m = torch.mul(hm_truth, hm_mask)

    # Forward Model
    pafA, pafB, pafC, hm = model.forward(imgs)

    # Opt
    loss = 0
    loss += mseLoss(torch.mul(pafA, paf_mask), paf_truth_m)
    loss += mseLoss(torch.mul(pafB, paf_mask), paf_truth_m)
    loss += mseLoss(torch.mul(pafC, paf_mask), paf_truth_m)
    loss += mseLoss(torch.mul(hm, hm_mask), hm_truth_m)

    # Opt
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Save every 2k
    if iterations % 2000 == 0 or exit:
        print("Saving")
        save_checkpoint({
            'iterations': iterations,
            'state_dict': model.state_dict(),
        }, NAME)
    if exit:
        print("Exiting..")
        control.value = 0
        sys.exit()
    print((iterations,loss))

    # OP Test
    if int(args.debug):
        test_index = 0
        hm_final = hm[test_index,:,:,:]
        paf_final = pafC[test_index,:,:,:]
        poseHeatMaps = torch.cat([hm_final, paf_final], 0).detach().cpu().numpy().copy()
        imageToProcess = imgs.detach().cpu().numpy().copy()[test_index,:,:,:]
        imageToProcess = (cv2.merge([imageToProcess[0,:,:]+0.5, imageToProcess[1,:,:]+0.5, imageToProcess[2,:,:]+0.5])*255).astype(np.uint8)
        datum = op.Datum()
        datum.cvInputData = imageToProcess
        datum.poseNetOutput = poseHeatMaps
        opWrapper.emplaceAndPop([datum])
        #print("Body keypoints: \n" + str(datum.poseKeypoints))
        cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData)
        cv2.waitKey(100)

    # img_viz = imgs.detach().cpu().numpy().copy()[0,0,:,:]
    # hm_pred_viz = hm.detach().cpu().numpy().copy()[0,0,:,:]
    # hm_truth_viz = hm_truth_m.cpu().numpy().copy()[0,0,:,:]
    # cv2.imshow("hm_pred_viz", cv2.resize(hm_pred_viz, (0,0), fx=8, fy=8, interpolation = cv2.INTER_CUBIC))
    # cv2.imshow("hm_truth_viz", cv2.resize(hm_truth_viz, (0,0), fx=8, fy=8, interpolation = cv2.INTER_CUBIC))
    # cv2.imshow("img", img_viz+0.5)
    # cv2.waitKey(0)


"""
Training of POF?
"""