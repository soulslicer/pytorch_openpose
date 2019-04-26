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
from natsort import natsorted, ns
import glob
dir_path = os.path.dirname(os.path.realpath(__file__))
import caffe
import json

def load_json(path):
    if path.endswith(".json"):
        with open(path) as json_data:
            #print path
            d = json.load(json_data)
            json_data.close()
            return d
    return 0

def pad_image(image, padValue, bbox):
    h = image.shape[0]
    h = min(bbox[0], h);
    w = image.shape[1]
    bbox[0] = (np.ceil(bbox[0]/8.))*8;
    bbox[1] = max(bbox[1], w);
    bbox[1] = (np.ceil(bbox[1]/8.))*8;
    pad = np.zeros(shape=(4))
    pad[0] = 0;
    pad[1] = 0;
    pad[2] = int(bbox[0]-h);
    pad[3] = int(bbox[1]-w);
    imagePadded = image
    padDown = np.tile(imagePadded[imagePadded.shape[0]-2:imagePadded.shape[0]-1,:,:], [int(pad[2]), 1, 1])*0
    imagePadded = np.vstack((imagePadded,padDown))
    padRight = np.tile(imagePadded[:,imagePadded.shape[1]-2:imagePadded.shape[1]-1,:], [1, int(pad[3]), 1])*0 + padValue
    imagePadded = np.hstack((imagePadded,padRight))
    return imagePadded, pad

def process_frame(frame, boxsize):
    height, width, channels = frame.shape
    scaleImage = float(boxsize) / float(height)
    rframe = cv2.resize(frame, (0,0), fx=scaleImage, fy=scaleImage)
    bbox = [boxsize, max(rframe.shape[1], boxsize)];
    imageForNet, padding = pad_image(rframe, 0, bbox)
    imageForNet = imageForNet.astype(np.float32)
    imageForNet = imageForNet/256. - 0.5
    imageForNet = np.transpose(imageForNet, (2,0,1))
    return rframe, imageForNet, scaleImage

TOTAL_FM = 128
TOTAL_PAFS = 72
TOTAL_POFS = 24*3
TOTAL_HMS = 25
ITERATIONS = 5

def save_checkpoint(state, directory):
    """Saves checkpoint to disk"""
    directory = dir_path + "/" + directory + "/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + "/" + str(state["iterations"]) + ".pth"
    torch.save(state, filename)

def load_checkpoint(directory):
    directory = dir_path + "/" + directory + "/"
    data = natsorted(glob.glob(directory + "*.pth"))
    if len(data) == 0: return None
    latest = data[-1]
    return torch.load(latest)

class ResConvBlock(torch.nn.Module):
    def __init__(self, In_D, Out_D):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResConvBlock, self).__init__()

        self.a = nn.Sequential(
            nn.Conv2d(In_D, Out_D, 3, 1, 1),
            nn.PReLU(Out_D)
        )
        self.b = nn.Sequential(
            nn.Conv2d(Out_D, Out_D, 3, 1, 1),
            nn.PReLU(Out_D)
        )
        self.c = nn.Sequential(
            nn.Conv2d(Out_D, Out_D, 3, 1, 1),
            nn.PReLU(Out_D)
        )

    def forward(self, x):
        a_output = self.a(x)
        b_output = self.b(a_output)
        c_output = self.c(b_output)
        output = torch.cat([a_output, b_output, c_output], 1)
        return output

class SConvBlock(torch.nn.Module):
    def __init__(self, In_D, Out_D):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ResConvBlock, self).__init__()

        self.a = nn.Conv2d(In_D, Out_D, 7, 1, 1)

    def forward(self, x):
        a_output = self.a(x)
        return output

class ABlock3x3(torch.nn.Module):
    def __init__(self, In_D, Out_D, Depth=64, SubDepth=256):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ABlock3x3, self).__init__()

        self.net = nn.Sequential(
            ResConvBlock(In_D, Depth),
            ResConvBlock(Depth*3, Depth),
            ResConvBlock(Depth*3, Depth),
            ResConvBlock(Depth*3, Depth),
            ResConvBlock(Depth*3, Depth),
            nn.Conv2d(Depth*3, SubDepth, 1, 1, 0),
            nn.PReLU(SubDepth),
            nn.Conv2d(SubDepth, Out_D, 1, 1, 0),
        )

    def forward(self, x):
        return self.net(x)

class ABlock3x3_Extended(torch.nn.Module):
    def __init__(self, In_D, Out_D, Depth=64, SubDepth=256):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ABlock3x3_Extended, self).__init__()

        self.net = nn.Sequential(
            ResConvBlock(In_D, Depth),
            ResConvBlock(Depth*3, Depth),
            ResConvBlock(Depth*3, Depth),
            ResConvBlock(Depth*3, Depth),
            ResConvBlock(Depth*3, Depth),
            ResConvBlock(Depth*3, Depth),
            ResConvBlock(Depth*3, Depth),
            ResConvBlock(Depth*3, Depth),
            nn.Conv2d(Depth*3, SubDepth, 1, 1, 0),
            nn.PReLU(SubDepth),
            nn.Conv2d(SubDepth, Out_D, 1, 1, 0),
        )

    def forward(self, x):
        return self.net(x)

class ABlock7x7(torch.nn.Module):
    def __init__(self, In_D, Out_D, Depth=64):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(ABlock7x7, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(In_D, Depth, 7, 1, 3),
            nn.ReLU(),
            nn.Conv2d(Depth, Depth, 7, 1, 3),
            nn.ReLU(),
            nn.Conv2d(Depth, Depth, 7, 1, 3),
            nn.ReLU(),
            nn.Conv2d(Depth, Depth, 7, 1, 3),
            nn.ReLU(),
            nn.Conv2d(Depth, Depth, 7, 1, 3),
            nn.ReLU(),
            nn.Conv2d(Depth, 256, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(256, Out_D, 1, 1, 0),
        )

    def forward(self, x):
        return self.net(x)

class Body25(nn.Module):

    def __init__(self, mode="3x3"):
        super(Body25, self).__init__()

        # Input Channel, Channel, Kernel Size, Stride, Padding
        self.vgg19 = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(3, 64, 3, 1, 1)),
            ('relu1_1', nn.ReLU()),
            ('conv1_2', nn.Conv2d(64, 64, 3, 1, 1)),
            ('relu1_2', nn.ReLU()),
            ('pool1_stage1', nn.MaxPool2d(2, 2)),

            ('conv2_1', nn.Conv2d(64, 128, 3, 1, 1)),
            ('relu2_1', nn.ReLU()),
            ('conv2_2', nn.Conv2d(128, 128, 3, 1, 1)),
            ('relu2_2', nn.ReLU()),
            ('pool2_stage1', nn.MaxPool2d(2, 2)),

            ('conv3_1', nn.Conv2d(128, 256, 3, 1, 1)),
            ('relu3_1', nn.ReLU()),
            ('conv3_2', nn.Conv2d(256, 256, 3, 1, 1)),
            ('relu3_2', nn.ReLU()),
            ('conv3_3', nn.Conv2d(256, 256, 3, 1, 1)),
            ('relu3_3', nn.ReLU()),
            ('conv3_4', nn.Conv2d(256, 256, 3, 1, 1)),
            ('relu3_4', nn.ReLU()),
            ('pool3_stage1', nn.MaxPool2d(2, 2)),

            ('conv4_1', nn.Conv2d(256, 512, 3, 1, 1)),
            ('relu4_1', nn.ReLU()),
            ('conv4_2', nn.Conv2d(512, 512, 3, 1, 1)),
            ('prelu4_2', nn.PReLU(512)),

            ('conv4_3_CPM', nn.Conv2d(512, 256, 3, 1, 1)),
            ('prelu4_3_CPM', nn.PReLU(256)),
            ('conv4_4_CPM', nn.Conv2d(256, TOTAL_FM, 3, 1, 1)),
            ('prelu4_4_CPM', nn.PReLU(128)),
        ]))

        if mode == "3x3":
            self.pafA = ABlock3x3(TOTAL_FM, TOTAL_PAFS, Depth=64, SubDepth=256)
            self.hmA = ABlock3x3(TOTAL_FM + TOTAL_PAFS, TOTAL_HMS, Depth=64, SubDepth=256)
            self.pafB = ABlock3x3(TOTAL_FM + TOTAL_PAFS, TOTAL_PAFS, Depth=128, SubDepth=512) #Fm, PrevPaf
            self.hmB = ABlock3x3(TOTAL_FM + TOTAL_HMS + TOTAL_PAFS, TOTAL_HMS, Depth=128, SubDepth=512) #Fm, PrevHm, CurrPaf
        elif mode == "7x7":
            self.pafA = ABlock7x7(TOTAL_FM, TOTAL_PAFS, Depth=64)
            self.hmA = ABlock7x7(TOTAL_FM + TOTAL_PAFS, TOTAL_HMS, Depth=64)
            self.pafB = ABlock7x7(TOTAL_FM + TOTAL_PAFS, TOTAL_PAFS, Depth=128) #Fm, PrevPaf
            self.hmB = ABlock7x7(TOTAL_FM + TOTAL_HMS + TOTAL_PAFS, TOTAL_HMS, Depth=128) #Fm, PrevHm, CurrPaf

        self.load_vgg()

    def load_vgg(self):
        # Apply Xavier Init
        def init_weights(m):
            if type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.pafA.apply(init_weights)
        self.hmA.apply(init_weights)
        self.pafB.apply(init_weights)
        self.hmB.apply(init_weights)

        # Load VGG19
        vgg19_model_url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
        vgg_state_dict = model_zoo.load_url(vgg19_model_url, model_dir=dir_path)  # save to local path
        vgg_keys = vgg_state_dict.keys()
        weights_load = {}
        for i in range(20):  # use first 10 conv layers of vgg19, 20 weights in all (weight+bias)
            weights_load[list(self.vgg19.state_dict().keys())[i]] = vgg_state_dict[list(vgg_keys)[i]]

        state = self.vgg19.state_dict()
        state.update(weights_load)
        self.vgg19.load_state_dict(state)

    def load_caffe_se(self, start_name, end_name, caffe_net, torch_net):
        weights_load = {}
        i = -1
        start = False
        for key in caffe_net.params:
            if key == start_name: start = True
            if not start: continue

            i+=1
            W = caffe_net.params[key][0].data[...]
            weights_load[torch_net.state_dict().keys()[i]] = torch.tensor(W)
            if len(caffe_net.params[key]) > 1:
                i+=1
                b = caffe_net.params[key][1].data[...]
                weights_load[torch_net.state_dict().keys()[i]] = torch.tensor(b)

            if key == end_name: start = False

        return weights_load

    def load_caffe(self):
        caffe_net = caffe.Net('/media/raaj/Storage/openpose_train/training_results_gines_new/pose_image/pose_deploy.prototxt',
                        '/media/raaj/Storage/openpose_train/training_results_gines_new/pose_image/model/pose_iter_608000.caffemodel',
                        caffe.TEST)

        # Load VGG19
        weights_load = {}
        for key in caffe_net.params:
            W = caffe_net.params[key][0].data[...]
            weights_load[key+"."+"weight"] = torch.tensor(W)
            if len(caffe_net.params[key]) > 1:
                b = caffe_net.params[key][1].data[...]
                weights_load[key+"."+"bias"] = torch.tensor(b)
            if key == "prelu4_4_CPM": 
                break
        state = self.vgg19.state_dict()
        state.update(weights_load)
        self.vgg19.load_state_dict(state)

        # Paf A
        weights_load = self.load_caffe_se("Mconv1_stage0_L2_0", "Mconv7_stage0_L2", caffe_net, self.pafA)
        state = self.pafA.state_dict()
        state.update(weights_load)
        self.pafA.load_state_dict(state)

        # Hm A
        weights_load = self.load_caffe_se("Mconv1_stage0_L1_0", "Mconv7_stage0_L1", caffe_net, self.hmA)
        state = self.hmA.state_dict()
        state.update(weights_load)
        self.hmA.load_state_dict(state)

        # Paf B
        weights_load = self.load_caffe_se("Mconv1_stage1_L2_0", "Mconv7_stage1_L2", caffe_net, self.pafB)
        state = self.pafB.state_dict()
        state.update(weights_load)
        self.pafB.load_state_dict(state)

        # Hm B
        weights_load = self.load_caffe_se("Mconv1_stage2_L1_0", "Mconv7_stage2_L1", caffe_net, self.hmB)
        state = self.hmB.state_dict()
        state.update(weights_load)
        self.hmB.load_state_dict(state)

    def forward(self, input):
        vgg_out = self.vgg19(input)

        paf_out = self.pafA(vgg_out)
        hm_out = self.hmA(torch.cat([vgg_out, paf_out], 1))

        prev_pafs = [paf_out]
        prev_hms = [hm_out]

        for i in range(0, ITERATIONS):
            paf_out = self.pafB(torch.cat([vgg_out, prev_pafs[-1]], 1))
            hm_out = self.hmB(torch.cat([vgg_out, prev_hms[-1], paf_out], 1))
            prev_pafs.append(paf_out)
            prev_hms.append(hm_out)

        return prev_pafs, prev_hms

class Model(nn.Module):
    def __init__(self, model, ngpu = 1):
        super(Model, self).__init__()
        self.ngpu = ngpu

        self.net = model

    def forward(self, input, mode=False):
        batch_size = input.size(0)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, input, range(self.ngpu))
        else:
            output = self.net(input)
        return output


class Gines(nn.Module):

    def __init__(self, pof=False):
        super(Gines, self).__init__()
        self.pof = pof

        # Input Channel, Channel, Kernel Size, Stride, Padding
        self.vgg19 = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(3, 64, 3, 1, 1)),
            ('relu1_1', nn.ReLU()),
            ('conv1_2', nn.Conv2d(64, 64, 3, 1, 1)),
            ('relu1_2', nn.ReLU()),
            ('pool1_stage1', nn.MaxPool2d(2, 2)),

            ('conv2_1', nn.Conv2d(64, 128, 3, 1, 1)),
            ('relu2_1', nn.ReLU()),
            ('conv2_2', nn.Conv2d(128, 128, 3, 1, 1)),
            ('relu2_2', nn.ReLU()),
            ('pool2_stage1', nn.MaxPool2d(2, 2)),

            ('conv3_1', nn.Conv2d(128, 256, 3, 1, 1)),
            ('relu3_1', nn.ReLU()),
            ('conv3_2', nn.Conv2d(256, 256, 3, 1, 1)),
            ('relu3_2', nn.ReLU()),
            ('conv3_3', nn.Conv2d(256, 256, 3, 1, 1)),
            ('relu3_3', nn.ReLU()),
            ('conv3_4', nn.Conv2d(256, 256, 3, 1, 1)),
            ('relu3_4', nn.ReLU()),
            ('pool3_stage1', nn.MaxPool2d(2, 2)),

            ('conv4_1', nn.Conv2d(256, 512, 3, 1, 1)),
            ('relu4_1', nn.ReLU()),
            ('conv4_2', nn.Conv2d(512, 512, 3, 1, 1)),
            ('prelu4_2', nn.PReLU(512)),

            ('conv4_3_CPM', nn.Conv2d(512, 256, 3, 1, 1)),
            ('prelu4_3_CPM', nn.PReLU(256)),
            ('conv4_4_CPM', nn.Conv2d(256, TOTAL_FM, 3, 1, 1)),
            ('prelu4_4_CPM', nn.PReLU(128)),
        ]))

        self.pafA = ABlock3x3_Extended(TOTAL_FM, TOTAL_PAFS, Depth=96, SubDepth=256)
        self.pafB = ABlock3x3_Extended(TOTAL_FM + TOTAL_PAFS, TOTAL_PAFS, Depth=256, SubDepth=512)
        self.pafC = ABlock3x3_Extended(TOTAL_FM + TOTAL_PAFS, TOTAL_PAFS, Depth=256, SubDepth=512)
        self.hmNetwork = ABlock3x3_Extended(TOTAL_FM + TOTAL_PAFS, TOTAL_HMS, Depth=192, SubDepth=512)

        # POF Networks
        if self.pof:
            self.pofA = ABlock3x3_Extended(TOTAL_FM + TOTAL_PAFS, TOTAL_POFS, Depth=128, SubDepth=256)
            self.pofB = ABlock3x3_Extended(TOTAL_FM + TOTAL_PAFS + TOTAL_POFS, TOTAL_POFS, Depth=256, SubDepth=512)

        self.load_vgg()

    def load_vgg(self):
        # Apply Xavier Init
        def init_weights(m):
            if type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        self.pafA.apply(init_weights)
        self.pafB.apply(init_weights)
        self.pafC.apply(init_weights)
        self.hmNetwork.apply(init_weights)
        if self.pof:
            self.pofA.apply(init_weights)
            self.pofB.apply(init_weights)            

        # Load VGG19
        vgg19_model_url = 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'
        vgg_state_dict = model_zoo.load_url(vgg19_model_url, model_dir=dir_path)  # save to local path
        vgg_keys = vgg_state_dict.keys()
        weights_load = {}
        for i in range(20):  # use first 10 conv layers of vgg19, 20 weights in all (weight+bias)
            weights_load[list(self.vgg19.state_dict().keys())[i]] = vgg_state_dict[list(vgg_keys)[i]]

        state = self.vgg19.state_dict()
        state.update(weights_load)
        self.vgg19.load_state_dict(state)

    def load_caffe_se(self, start_name, end_name, caffe_net, torch_net):
        weights_load = {}
        i = -1
        start = False
        for key in caffe_net.params:
            if key == start_name: start = True
            if not start: continue

            i+=1
            W = caffe_net.params[key][0].data[...]
            weights_load[torch_net.state_dict().keys()[i]] = torch.tensor(W)
            if len(caffe_net.params[key]) > 1:
                i+=1
                b = caffe_net.params[key][1].data[...]
                weights_load[torch_net.state_dict().keys()[i]] = torch.tensor(b)

            if key == end_name: start = False

        return weights_load

    def load_caffe(self):
        caffe_net = caffe.Net('/media/posefs3b/Users/gines/openpose_train/training_results/5_25BSuperModel31DeepAndHM/best_586k/pose/body_25b/pose_deploy.prototxt',
                        '/media/posefs3b/Users/gines/openpose_train/training_results/5_25BSuperModel31DeepAndHM/best_586k/pose/body_25b/pose_iter_586000.caffemodel',
                        caffe.TEST)

        # Load VGG19
        weights_load = {}
        for key in caffe_net.params:
            W = caffe_net.params[key][0].data[...]
            weights_load[key+"."+"weight"] = torch.tensor(W)
            if len(caffe_net.params[key]) > 1:
                b = caffe_net.params[key][1].data[...]
                weights_load[key+"."+"bias"] = torch.tensor(b)
            if key == "prelu4_4_CPM": 
                break
        state = self.vgg19.state_dict()
        state.update(weights_load)
        self.vgg19.load_state_dict(state)

        # Paf A
        weights_load = self.load_caffe_se("Mconv1_stage0_L2_0", "Mconv10_stage0_L2", caffe_net, self.pafA)
        state = self.pafA.state_dict()
        state.update(weights_load)
        self.pafA.load_state_dict(state)

        # Paf B
        weights_load = self.load_caffe_se("Mconv1_stage1_L2_0", "Mconv10_stage1_L2", caffe_net, self.pafB)
        state = self.pafB.state_dict()
        state.update(weights_load)
        self.pafB.load_state_dict(state)

        # Paf C
        weights_load = self.load_caffe_se("Mconv1_stage2_L2_0", "Mconv10_stage2_L2", caffe_net, self.pafC)
        state = self.pafC.state_dict()
        state.update(weights_load)
        self.pafC.load_state_dict(state)

        # Hm B
        weights_load = self.load_caffe_se("Mconv1_stage0_L1_0", "Mconv10_stage0_L1", caffe_net, self.hmNetwork)
        state = self.hmNetwork.state_dict()
        state.update(weights_load)
        self.hmNetwork.load_state_dict(state)

    def forward(self, input):
        vgg_out = self.vgg19(input)

        pafA = self.pafA(vgg_out)
        pafB = self.pafB(torch.cat([vgg_out, pafA], 1))
        pafC = self.pafC(torch.cat([vgg_out, pafB], 1))
        hm = self.hmNetwork(torch.cat([vgg_out, pafC], 1))

        if self.pof:

            pofA = self.pofA(torch.cat([vgg_out, pafC], 1))
            pofB = self.pofB(torch.cat([vgg_out, pafC, pofA], 1))

            return pafA, pafB, pafC, hm, pofA, pofB

        else:

            return pafA, pafB, pafC, hm
