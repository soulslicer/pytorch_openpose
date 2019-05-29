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

# Params
NAME = "weights_op"
OP_CAFFE_TRAIN_PATH = '/home/raaj/openpose_caffe_train/build/op/'
OP_PYTHON_PATH = '/home/raaj/openpose_orig/build/python/'
OP_MODEL_FOLDER = '/home/raaj/openpose_orig/models/'
OP_LMDB_FOLDER = '/media/raaj/Storage/openpose_train/dataset/'

# Import OP
import sys
sys.path.insert(0, OP_CAFFE_TRAIN_PATH)
import opcaffe
import signal
sys.path.append(OP_PYTHON_PATH)
from openpose import pyopenpose as op
from models import *

# DATA
VAL_PATH = dir_path + "/val2017/*.jpg"
COCOSAVE_PATH = dir_path + "/coco_result.json"

# Parsers
parser = argparse.ArgumentParser(description='OP')
parser.add_argument('--weight', type=str, default="",
                    help='Weight')
args = parser.parse_args()


# Sample OP Network
params = dict()
params["model_folder"] = OP_MODEL_FOLDER
params["body"] = 2  # Disable OP Network
params["upsampling_ratio"] = 0
params["model_pose"] = "BODY_25B"
params["write_coco_json"] = COCOSAVE_PATH
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# Setup Model
model = Model(Body25B(), ngpu=int(1)).cuda()
model.eval()

# Load weights
if len(args.weight):
    state = torch.load(args.weight)
    if state != None:
        model.load_state_dict(state['state_dict'])
        print("Loaded State")
else:
    model.net.load_caffe()

model.net.load_caffe()

# Serialize
example = torch.rand(1, 3, 368, 368).cuda()
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save(NAME+"/"+"op_model.pt")
traced_script_module = torch.jit.load(NAME+"/"+"op_model.pt")

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
    #print(scaleFactor)

    # Model
    pafA, pafB, pafC, pafD, paf_final, hm_final = traced_script_module(imageForNet)
    #pafA, pafB, pafC, pafD, paf_final, hm_final = model.forward(imageForNet)

    # OP Test
    test_index = 0
    poseHeatMaps = torch.cat([hm_final, paf_final], 1).detach().cpu().numpy().copy()
    imageToProcess = imageForNet.detach().cpu().numpy().copy()[test_index,:,:,:]
    imageToProcess = (cv2.merge([imageToProcess[0,:,:]+0.5, imageToProcess[1,:,:]+0.5, imageToProcess[2,:,:]+0.5])*255).astype(np.uint8)
    datum = op.Datum()
    datum.name = true_name
    datum.cvInputData = imageToProcess
    datum.poseNetOutput = poseHeatMaps
    opWrapper.emplaceAndPop([datum])

    print(datum.poseKeypoints.shape)
    print("Body keypoints: \n" + str(datum.poseKeypoints))
    cv2.imshow("OpenPose 1.4.0 - Tutorial Python API", datum.cvOutputData)
    cv2.waitKey(0)

# Stop
opWrapper.stop()
del opWrapper

# Load JSON
json_result = load_json(COCOSAVE_PATH)
for item in json_result:
    sf = scale_factors[int(item["image_id"])]
    for i in range(0, len(item["keypoints"])):
        if i % 3 != 0:
            true_index = i-1
            item["keypoints"][true_index] /= sf

with open(COCOSAVE_PATH, 'w') as fp:
    json.dump(json_result, fp)