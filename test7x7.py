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


# Parsers
parser = argparse.ArgumentParser(description='OP')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')
parser.add_argument('--batch', type=int, default=10,
                    help='batch size')
parser.add_argument('--reload', action='store_true')
args = parser.parse_args()


# DATA
VAL_PATH = dir_path + "/val2017/*.jpg"
COCOSAVE_PATH = dir_path + "/coco_result.json"

# Sample OP Network
params = dict()
params["model_folder"] = "/home/raaj/openpose_orig/models/"
params["body"] = 2  # Disable OP Network
params["upsampling_ratio"] = 0
params["model_pose"] = "BODY_25B"
params["write_coco_json"] = COCOSAVE_PATH
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Setup Model
NAME = "weights7x7"
model = Model(Body25("7x7"), ngpu=int(1)).cuda()
model.eval()

# Load Weights
iterations = 0
reload = int(args.reload)
if not reload:
    state = load_checkpoint(NAME)
    if state != None:
        iterations = state["iterations"]
        model.load_state_dict(state['state_dict'])
        print("Loaded Iteration " + str(iterations))

# Validation Location
iterations = -1
import glob
import natsort
image_files = natsort.natsorted(glob.glob(VAL_PATH))
scale_factors = dict()
for image_file in image_files:
    iterations += 1
    print(float(iterations)/float(len(image_files)))
    true_name = (image_file.split("/")[-1]).split(".")[0]
    img = cv2.imread(image_file)
    rframe, imageForNet, scaleFactor = process_frame(img, 368)
    imageForNet = torch.tensor(np.expand_dims(imageForNet, axis=0)).cuda()
    scale_factors[int(true_name)] = scaleFactor

    # Model
    pafs_pred, hms_pred = model.forward(imageForNet)

    # OP Test
    test_index = 0
    hm_final = hms_pred[ITERATIONS-1][test_index,:,:,:]
    paf_final = pafs_pred[ITERATIONS-1][test_index,:,:,:]
    poseHeatMaps = torch.cat([hm_final, paf_final], 0).detach().cpu().numpy().copy()
    imageToProcess = imageForNet.detach().cpu().numpy().copy()[test_index,:,:,:]
    imageToProcess = (cv2.merge([imageToProcess[0,:,:]+0.5, imageToProcess[1,:,:]+0.5, imageToProcess[2,:,:]+0.5])*255).astype(np.uint8)
    datum = op.Datum()
    datum.name = true_name
    datum.cvInputData = imageToProcess
    datum.poseNetOutput = poseHeatMaps
    opWrapper.emplaceAndPop([datum])
    #print(datum.poseKeypoints.shape)
    #print("Body keypoints: \n" + str(datum.poseKeypoints))
    cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData)
    cv2.waitKey(0)

# Stop
opWrapper.stop()
del opWrapper

# Load JSON
json_result = load_json("coco_result.json")
for item in json_result:
    sf = scale_factors[int(item["image_id"])]
    for i in range(0, len(item["keypoints"])):
        if i % 3 != 0:
            true_index = i-1
            item["keypoints"][true_index] /= sf

with open('coco_result.json', 'w') as fp:
    json.dump(json_result, fp)


 # Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.389
 # Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.649
 # Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.389
 # Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.324
 # Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.478
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.441
 # Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.673
 # Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.447
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.344
 # Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.576
