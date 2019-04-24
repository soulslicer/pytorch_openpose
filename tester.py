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
from natsort import natsorted, ns
import glob
import subprocess

PATH = "/home/raaj/disk/pytorch_openpose/weights_gines/"

paths = natsorted(glob.glob(PATH + "*.pth"))
for path in paths:
    print(path)

    command = "python test_gines.py --weight " + path
    # os.system(command)

    output = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()
    output = subprocess.Popen("python coco_eval.py", shell=True, stdout=subprocess.PIPE).stdout.read()

    output = output.split("\n")
    precision = output[-11].split(" = ")[-1]
    recall = output[-6].split(" = ")[-1]
    print(precision, recall)

    #break

