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

PATH = "/home/raaj/disk/pytorch_openpose/weights_gines_no/"

paths = natsorted(glob.glob(PATH + "*.pth"))
for path in paths:
    #path = "/home/raaj/disk/pytorch_openpose/weights_gines/88000.pth"
    print(path)

    command = "python test_gines.py --weight " + path

    output = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()
    output = subprocess.Popen("python coco_eval.py", shell=True, stdout=subprocess.PIPE).stdout.read()

    output = output.split("\n")
    precision = output[-11].split(" = ")[-1]
    recall = output[-6].split(" = ")[-1]
    print(precision, recall)

    #break

    """
    I need to debug the loader. Make both normal mode, and other mode save data to disk and visualize it
    """

    ###########

# 0.062 0.083
# raaj@raaj:~/Desktop/op_pytorch$ python tester.py 
# /home/raaj/disk/pytorch_openpose/weights_gines/2000.pth
# 0.062 0.083
# /home/raaj/disk/pytorch_openpose/weights_gines/4000.pth
# 0.106 0.141
# /home/raaj/disk/pytorch_openpose/weights_gines/6000.pth
# 0.151 0.187
# /home/raaj/disk/pytorch_openpose/weights_gines/8000.pth
# 0.191 0.236
# /home/raaj/disk/pytorch_openpose/weights_gines/10000.pth
# 0.202 0.251
# /home/raaj/disk/pytorch_openpose/weights_gines/12000.pth
# 0.230 0.271
# /home/raaj/disk/pytorch_openpose/weights_gines/14000.pth
# 0.231 0.267
# /home/raaj/disk/pytorch_openpose/weights_gines/16000.pth
# 0.242 0.286
# /home/raaj/disk/pytorch_openpose/weights_gines/18000.pth
# 0.270 0.310
# /home/raaj/disk/pytorch_openpose/weights_gines/20000.pth
# 0.266 0.314
# /home/raaj/disk/pytorch_openpose/weights_gines/22000.pth
# 0.270 0.315
# /home/raaj/disk/pytorch_openpose/weights_gines/24000.pth
# 0.280 0.324
# /home/raaj/disk/pytorch_openpose/weights_gines/26000.pth
# 0.286 0.338
# /home/raaj/disk/pytorch_openpose/weights_gines/28000.pth
# 0.293 0.339
# /home/raaj/disk/pytorch_openpose/weights_gines/30000.pth
# 0.279 0.331
# /home/raaj/disk/pytorch_openpose/weights_gines/32000.pth
# 0.296 0.348
# /home/raaj/disk/pytorch_openpose/weights_gines/34000.pth
# 0.302 0.350
# /home/raaj/disk/pytorch_openpose/weights_gines/36000.pth
# 0.312 0.357
# /home/raaj/disk/pytorch_openpose/weights_gines/38000.pth
# 0.312 0.366
# /home/raaj/disk/pytorch_openpose/weights_gines/40000.pth
# 0.311 0.363
# /home/raaj/disk/pytorch_openpose/weights_gines/42000.pth
# 0.320 0.362
# /home/raaj/disk/pytorch_openpose/weights_gines/44000.pth
# 0.321 0.368
# /home/raaj/disk/pytorch_openpose/weights_gines/46000.pth
# 0.321 0.370
# /home/raaj/disk/pytorch_openpose/weights_gines/48000.pth
# 0.325 0.371
# /home/raaj/disk/pytorch_openpose/weights_gines/50000.pth
# 0.320 0.373
# /home/raaj/disk/pytorch_openpose/weights_gines/52000.pth
# 0.336 0.384
# /home/raaj/disk/pytorch_openpose/weights_gines/54000.pth
# 0.336 0.391
# /home/raaj/disk/pytorch_openpose/weights_gines/56000.pth
# 0.336 0.380
# /home/raaj/disk/pytorch_openpose/weights_gines/58000.pth
# 0.336 0.383
# /home/raaj/disk/pytorch_openpose/weights_gines/60000.pth
# 0.340 0.387
# /home/raaj/disk/pytorch_openpose/weights_gines/62000.pth
# 0.345 0.391
# /home/raaj/disk/pytorch_openpose/weights_gines/64000.pth
# 0.353 0.401
# /home/raaj/disk/pytorch_openpose/weights_gines/66000.pth
# 0.342 0.387
# /home/raaj/disk/pytorch_openpose/weights_gines/68000.pth
# 0.357 0.404
# /home/raaj/disk/pytorch_openpose/weights_gines/70000.pth
# 0.352 0.404
# /home/raaj/disk/pytorch_openpose/weights_gines/72000.pth
# 0.355 0.401
# /home/raaj/disk/pytorch_openpose/weights_gines/74000.pth
# 0.360 0.413
# /home/raaj/disk/pytorch_openpose/weights_gines/76000.pth
# 0.360 0.405
# /home/raaj/disk/pytorch_openpose/weights_gines/78000.pth
# 0.358 0.408
# /home/raaj/disk/pytorch_openpose/weights_gines/80000.pth
# 0.363 0.412
# /home/raaj/disk/pytorch_openpose/weights_gines/82000.pth
# 0.355 0.407
# /home/raaj/disk/pytorch_openpose/weights_gines/84000.pth
# 0.351 0.395
# /home/raaj/disk/pytorch_openpose/weights_gines/86000.pth
# 0.349 0.397
# /home/raaj/disk/pytorch_openpose/weights_gines/88000.pth
# 0.358 0.402
# /home/raaj/disk/pytorch_openpose/weights_gines/88713.pth
# 0.355 0.406
# raaj@raaj:~/Desktop/op_pytorch$ python tester.py 

