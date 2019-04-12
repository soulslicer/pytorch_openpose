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
from triplet_image_loader import TripletImageLoader
import scipy.io as sio
import time

################################################
# insert this to the top of your scripts (usually main.py)
# This is due to updated PyTorch
################################################
import sys, warnings, traceback, torch
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
    traceback.print_stack(sys._getframe(2))
warnings.showwarning = warn_with_traceback; warnings.simplefilter('always', UserWarning);
torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

def weights_init(m):
    '''
    Custom weights initialization called on encoder and decoder.
    '''
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data, a=0.01)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, std=0.015)
        m.bias.data.zero_()

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * ((1 - 0.015) ** epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def half_lr(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] /= 2.

def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (1.0* int((pred > 0).sum().item()))/ (1.0* dista.size()[0])

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

import numpy as np

################################################
### Training settings
### These are different parameters for model/data/hyperparameter 
### The details for each can be found in "help = ...." descriptions
################################################

# that can be set while running the script from the terminal.
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--train_batch_size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 64)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 5e-5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='Conditional_Similarity_Network', type=str,
                    help='name of experiment')
parser.add_argument('--embed_loss', type=float, default=5e-3, metavar='M',
                    help='parameter for loss for embedding norm')
parser.add_argument('--vae_loss', type=float, default=1, metavar='M',
                    help='parameter for loss for embedding norm')
parser.add_argument('--triplet_loss', type=float, default=1, metavar='M',
                    help='parameter for loss for embedding norm')
parser.add_argument('--mask_loss', type=float, default=5e-4, metavar='M',
                    help='parameter for loss for mask norm')
parser.add_argument('--num_traintriplets', type=int, default=50000, metavar='N',
                    help='how many unique training triplets (default: 50000)')
parser.add_argument('--num_valtriplets', type=int, default=20000, metavar='N',
                    help='how many unique validation triplets (default: 10000)')
parser.add_argument('--num_testtriplets', type=int, default=40000, metavar='N',
                    help='how many unique test triplets (default: 20000)')

parser.add_argument('--dim_embed', type=int, default=64, metavar='N',
                    help='how many dimensions in embedding (default: 64)')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--learned', dest='learned', action='store_true',
                    help='To learn masks from random initialization')
parser.add_argument('--prein', dest='prein', action='store_true',
                    help='To initialize masks to be disjoint')
parser.add_argument('--visdom', dest='visdom', action='store_true',
                    help='Use visdom to track and plot')

parser.add_argument('--image_size', type=int, default=224,  
                    help='height/width length of the input images, default=64')

parser.add_argument('--ndf', type=int, default=32,
                    help='number of output channels for the first decoder layer, default=32')

parser.add_argument('--nef', type=int, default=32,
                    help='number of output channels for the first encoder layer, default=32')

#same as dim_embed
parser.add_argument('--nz', type=int, default=128,
                    help='size of the latent vector z, default=64')
parser.add_argument('--ngpu', type=int, default=1,
                    help='number of GPUs to use')

parser.add_argument('--outf', default='./output',
                    help='folder to output images and model checkpoints')


parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam, default=0.1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='beta2 for adam, default=0.001')


parser.add_argument('--nc', type=int, default=1,
                    help='number of input channel in data. 3 for rgb, 1 for grayscale')
parser.set_defaults(test=False)
parser.set_defaults(learned=False)
parser.set_defaults(prein=False)
parser.set_defaults(visdom=False)

best_acc = 0

class _Encoder(nn.Module):

    def __init__(self, ngpu,nc,nef,out_size,nz):
        super(_Encoder, self).__init__()
        self.ngpu = ngpu
        self.nc = nc 
        self.nef = nef
        self.out_size = out_size
        self.nz = nz
        self.encoder = nn.Sequential(
            # Input Channel, Channel, Kernel Size, Stride, Padding
            # nn.Conv2d(nc, nef, 4, 2, padding=1),
            # nn.BatchNorm2d(nef),
            # nn.LeakyReLU(0.2, True),

            # nn.Conv2d(nef, nef * 2, 4, 2, padding=1),
            # nn.BatchNorm2d(nef*2),
            # nn.LeakyReLU(0.2, True),

            # nn.Conv2d(nef * 2, nef * 4, 4, 2, padding=1),
            # nn.BatchNorm2d(nef*4),
            # nn.LeakyReLU(0.2, True),            

            # nn.Conv2d(nef * 4, nef * 8, 4, 2, padding=1),
            # nn.BatchNorm2d(nef*8),
            # nn.LeakyReLU(0.2, True),

            #### VGG 11 Style

            # 3 -> 32
            nn.Conv2d(nc, nef, 3, stride=1, padding=1),
            nn.BatchNorm2d(nef),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2),

            # 32 -> 64
            nn.Conv2d(nef, nef * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(nef*2),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2),

            # 64 -> 128
            nn.Conv2d(nef * 2, nef * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(nef*4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nef * 4, nef * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(nef*4),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2),

            # 128 -> 256
            nn.Conv2d(nef * 4, nef * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(nef*8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nef * 8, nef * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(nef*8),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2),

            # 256 -> 256 twice
            nn.Conv2d(nef * 8, nef * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(nef*8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nef * 8, nef * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(nef*8),
            nn.LeakyReLU(0.2, True),

            
        )
        # 8 because..the depth went from 32 to 32*8
        self.mean = nn.Linear(nef * 8 * out_size * (out_size/2), nz)
        self.logvar = nn.Linear(nef * 8 * out_size * (out_size/2), nz)

    #for reparametrization trick 
    def sampler(self, mean, logvar):  
        std = logvar.mul(0.5).exp_()
        if args.cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, input):
        batch_size = input.size(0)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            hidden = nn.parallel.data_parallel(
                self.encoder, input, range(self.ngpu))
            hidden = hidden.view(batch_size, -1)
            mean = nn.parallel.data_parallel(
                self.mean, hidden, range(self.ngpu))
            logvar = nn.parallel.data_parallel(
                self.logvar, hidden, range(self.ngpu))
        else:
            hidden = self.encoder(input)
            hidden = hidden.view(batch_size, -1)
            mean, logvar = self.mean(hidden), self.logvar(hidden)
        latent_z = self.sampler(mean, logvar)
        return latent_z,mean,logvar

class Tripletnet(nn.Module):
    def __init__(self, embeddingnet):
        super(Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet

    def forward(self, x, y, z):
        latent_x,mean_x,logvar_x = self.embeddingnet(x)
        latent_y,mean_y,logvar_y = self.embeddingnet(y)
        latent_z,mean_z,logvar_z = self.embeddingnet(z)
        dist_a = F.pairwise_distance(mean_x, mean_y, 2)
        dist_b = F.pairwise_distance(mean_x, mean_z, 2)
        return latent_x,mean_x,logvar_x,\
            latent_y,mean_y,logvar_y,\
            latent_z,mean_z,logvar_z,\
            dist_a, dist_b

class _Decoder(nn.Module):


    def __init__(self, ngpu,nc,ndf,out_size,nz):
        super(_Decoder, self).__init__()
        self.ngpu = ngpu
        self.nc = nc
        self.nz  = nz
        self.ndf = ndf
        self.out_size = out_size

        self.decoder_dense = nn.Sequential(
            nn.Linear(nz, ndf * 8 * out_size * out_size/2),
            nn.ReLU(True)
        )
        self.decoder_conv = nn.Sequential(
            # # Input Channel, Channel, Kernel Size, Stride, Padding
            # nn.Upsample(scale_factor=2,mode='nearest'),
            # nn.Conv2d(ndf * 8, ndf * 4, 3, padding=1),
            # nn.BatchNorm2d(ndf * 4, 1e-3),
            # nn.LeakyReLU(0.2, True),

            # nn.Upsample(scale_factor=2,mode='nearest'),
            # nn.Conv2d(ndf * 4, ndf * 2, 3, padding=1),
            # nn.BatchNorm2d(ndf * 2, 1e-3),
            # nn.LeakyReLU(0.2, True),
            

            # nn.Upsample(scale_factor=2,mode='nearest'),
            # nn.Conv2d(ndf * 2, ndf, 3, padding=1),
            # nn.BatchNorm2d(ndf, 1e-3),
            # nn.LeakyReLU(0.2, True),
            
            # nn.Upsample(scale_factor=2,mode='nearest'),
            # nn.Conv2d(ndf, nc, 3, padding=1)

            # VGG11
            nn.Conv2d(ndf * 8, ndf * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(ndf*8, 1e-3),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 8, ndf * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(ndf*8, 1e-3),
            nn.LeakyReLU(0.2, True),

            # 256 -> 128
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(ndf * 8, ndf * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(ndf*4, 1e-3),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(ndf*4, 1e-3),
            nn.LeakyReLU(0.2, True),
            
            # 128 -> 64
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(ndf * 4, ndf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ndf*2, 1e-3),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, ndf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ndf*2, 1e-3),
            nn.LeakyReLU(0.2, True),

            # 64 -> 32
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(ndf * 2, ndf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ndf, 1e-3),
            nn.LeakyReLU(0.2, True),

            # 32 -> 32
            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.Conv2d(ndf, nc, 3, stride=1, padding=1),


        )

    def forward(self, input):
        batch_size = input.size(0)
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            hidden = nn.parallel.data_parallel(
                self.decoder_dense, input, range(self.ngpu))
            hidden = hidden.view(batch_size, self.ndf * 8, self.out_size, self.out_size)
            output = nn.parallel.data_parallel(
                self.decoder_conv, input, range(self.ngpu))
        else:
            hidden = self.decoder_dense(input).view(
                batch_size, self.ndf * 8, self.out_size, self.out_size)
            output = self.decoder_conv(hidden)
        return output


def train(train_loader, tnet, decoder, criterion, optimizer, epoch):
    losses_metric = AverageMeter()
    losses_VAE = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()

    # Train
    tnet.train()
    decoder.train()
    for batch_idx, (anchor_out, pos_out, neg_out, anchor_in, pos_in, neg_in) in enumerate(train_loader):
        #print("Load Data")

        # Load into GPU
        anchor_out_var, pos_out_var, neg_out_var = Variable(anchor_out.cuda()), Variable(pos_out.cuda()), Variable(neg_out.cuda())
        anchor_in_var, pos_in_var, neg_in_var = Variable(anchor_in.cuda()), Variable(pos_in.cuda()), Variable(neg_in.cuda())

        # Compute Encoder
        latent_x,mean_x,logvar_x,latent_y,mean_y,logvar_y,latent_z,mean_z,logvar_z,dist_a, dist_b = tnet(anchor_in_var, pos_in_var, neg_in_var)
        
        # Compute Decoder
        reconstructed_x = decoder(latent_x)

        # (Apply Triplet loss) 1 means, dista should be larger than distb
        target = torch.FloatTensor(dist_a.size()).fill_(1)
        target = target.cuda()
        target = Variable(target)
        loss_triplet = criterion(dist_a, dist_b, target)
        loss_embedd = mean_x.norm(2) + mean_y.norm(2) + mean_z.norm(2)

        # Loss Combine
        loss = args.triplet_loss*loss_triplet + args.embed_loss*loss_embedd 
        # measure accuracy and record loss
        acc = accuracy(dist_a, dist_b)
        print(acc)
        losses_metric.update(loss_triplet.item(), anchor_in.size(0))
        #losses_VAE.update(loss_vae.data[0], data1.size(0))
        accs.update(acc, anchor_in.size(0))
        emb_norms.update(loss_embedd.item()/3, anchor_in.size(0))

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'VAE Loss: {:.4f} ({:.4f}) \t'
                  'Metric Loss: {:.4f} ({:.4f}) \t'
                  'Metric Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, batch_idx * len(anchor_in), len(train_loader.dataset),
                #losses_VAE.val, losses_VAE.avg,
                0,0,
                losses_metric.val, losses_metric.avg, 
                100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))

            train_loss_metric.append(losses_metric.val)
            #train_loss_VAE.append(losses_VAE.val)
            train_loss_VAE.append(0)
            train_acc_metric.append(accs.val)
        

def main():
    global args, best_acc
    global  log_interval
    log_interval = 30
    args = parser.parse_args()
    print(args)
    nz = int(args.dim_embed)
    nef = int(args.nef)
    ndf = int(args.ndf)
    ngpu = int(args.ngpu)
    nc = int(args.nc)
    out_size = args.image_size // 16

    # Data Loaders
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    normalize = transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                     std=[1, 1, 1])
    out_size = args.image_size // 16  
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        TripletImageLoader('data/FID-300/', [0,250],
                        transform=transforms.Compose([
                            transforms.Resize((224,112)),
                            transforms.CenterCrop((224,112)),
                            transforms.ToTensor(),
                            normalize,
                    ])),
        batch_size=args.train_batch_size, shuffle=True, **kwargs)

    # Encoder Network
    encoder = _Encoder(ngpu,nc,nef,out_size,nz)
    encoder.apply(weights_init)
    if args.cuda:
        encoder = encoder.cuda()
    tnet = Tripletnet(encoder)
    if args.cuda:
        tnet.cuda()

    # Decoder
    decoder = _Decoder(ngpu,nc,ndf,out_size,nz)
    decoder.apply(weights_init)

    # Global Storage
    global train_loss_metric,train_loss_VAE,train_acc_metric,test_loss_metric,test_loss_VAE,test_acc_metric
    train_loss_metric = []
    train_loss_VAE = []
    train_acc_metric = []
    test_loss_metric = []
    test_loss_VAE = []
    test_acc_metric = []

    # Loss Functions and Params
    criterion = torch.nn.MarginRankingLoss(margin = args.margin)
    parameters = list(tnet.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr, betas=(args.beta1, args.beta2))

    # Half points
    half_points = [50, 100, 150]

    for epoch in range(args.start_epoch, args.epochs + 1):
        # Half the LR based on above interval
        if epoch in half_points: 
            half_lr(optimizer)
            # Print
            print(epoch)
            print("LR: " + str(optimizer.param_groups[0]['lr']))

        # Train
        train(train_loader, tnet, decoder, criterion, optimizer, epoch)


    print("OK")

if __name__ == '__main__':
    main()    
