#__author__ = 'maoyeh'
#__version__ = '1.0'

import glob
import time
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torchvision

from torchvision import transforms
from collections import OrderedDict
from matplotlib.pyplot import imshow
import numpy as np
from scipy import interpolate
from scipy import misc
from PIL import Image
from torch.autograd import Variable
from shutil import copyfile





#vgg definition that conveniently let you grab the outputs from any layer
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
            
    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]
    
    
# a function generate corvariant matrix and means with input feature map tensor
class Cov_Mean(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b, c, h*w)
        mean_ = F.mean( dim=2, keepdim=True).detach()
        mean = torch.cat(h*w*[mean_], 2)
        F = F-mean.detach()
        G = torch.bmm(F, F.transpose(1,2)) 
        G.div_(h*w)
        return G.squeeze(0).data, mean_.squeeze().data 

    
    
# Our class of tools for generate base E statistics   
class TOOLS:   
    def __init__(self,model_dir,img_size,):
        
        #set parameters
        self.img_size = img_size
        self.PCA_basis = 0
        # Layers for style transfer
        self.style_layers = ['r11','r21','r31','r41', 'r51'] 
        self.content_layers = ['r42']  
        # Reduced Dimensions(ranks) for PCA
        self.Ks = [ 32,48,128,256,256]
        # pre  processing for images
        self.prep = transforms.Compose([transforms.Scale(img_size),
                                   transforms.ToTensor(),
                                   transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
                                   transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
                                                        std=[1,1,1]),
                                   transforms.Lambda(lambda x: x.mul_(255)),
                                  ])
        
        
        #get network
        self.vgg = VGG()
        self.vgg.load_state_dict(torch.load(model_dir + 'vgg_conv.pth'))
        for param in self.vgg.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            self.vgg.cuda()

        

    # project covariance matrix and mean (A) to PCA_basis (P) with reduced rank (k).
    def PCA_Proj(self,A,P,k):
        return torch.mm(torch.mm(P[0][:,:k].t(),A[0]),P[2][:,:k]), torch.mm( A[1].unsqueeze(0), P[0][:,:k] ) 
    
    # evaluate the log term in KL divergence formula
    def Det(self,A,B):
        _,S,_ = torch.svd(A)
        _,S1,_ = torch.svd(B)
        temp =torch.log(S1/S)
        u=0
        for a in temp:
            u +=a
        return u
    
    # given folder of reference images, generate a basis from svd decomposition of (averaged) covariance matrix of each layer 
    def PCA_Basis_Generater(self,Reference_dir):
        
        references=[]
        references = glob.glob(Reference_dir+"*.jpg")  # generate a list of reference images                    
                    
        Total_Covs=[0,0,0,0,0] # initialize a list of covariant matrices 5 layers 

        for sample in references:    

            img_torch = self.prep(Image.open(sample)) 

            if torch.cuda.is_available():
                img_torch = Variable(img_torch.unsqueeze(0).cuda()) 
            else:
                img_torch = Variable(img_torch.unsqueeze(0))
                
            Covs_Means = [Cov_Mean()(A) for A in self.vgg(img_torch, self.style_layers[:])] # generate corvariant matrix and means for each layer
            Total_Covs= [x+y[0] for x,y in zip(Total_Covs,Covs_Means)] # summation of corvariant matrix of each layer over references

        # Take an average over references
        AVG_Covs = [x/len(references) for x in Total_Covs]

        
        # make a decomposition (U, S, V) of each layer 
        self.PCA_basis = [torch.svd(data) for data in AVG_Covs]    

            
        return self.PCA_basis
    
    
    
    # generate Base E statistics of each sample image (source_dir, source_list) according to style images(style_dir) 
    # The formula is based on Multivariate normal distributions of https://en.wikipedia.org/wiki/Kullback–Leibler_divergence
    def E_Basic_Statistics(self, style_dir, source_dir, source_list, outputfile):
        # Reduced Dimensions(ranks) for PCA
        Ks = self.Ks # [ 32,48,128,256,256]  
        
        # setup for input and outpu files
        input_list = open(source_list, 'r') 
        output_list= open('%s%s'%(Ks,outputfile), 'w')
        columns = ["style", "content","weight","E1","E2","E3","E4","E5","\n"]
        name = '\t'.join(columns) 
        output_list.write(name)


        for line in input_list.readlines()[:]:
            
            ## with specific image file name on list (depending on filenames of source_list)
            filename  = line[:-1] 
            sp =line[:].split('_')
            style =int(sp[0][5:]) 
            content = int(sp[1][7:])
            print(style,content, filename)

            # pre-processing of images with typorch
            img_dirs = [style_dir,  source_dir]
            img_names = ['styles - %s.jpg'%style, filename]
            imgs = [Image.open(img_dirs[i] + name) for i,name in enumerate(img_names)]
            
            imgs_torch = [self.prep(img) for img in imgs]
            if torch.cuda.is_available():
                imgs_torch = [Variable(img.unsqueeze(0).cuda()) for img in imgs_torch]
            else:
                imgs_torch = [Variable(img.unsqueeze(0)) for img in imgs_torch]
            style_image,  syn_image = imgs_torch

            
            # generate convariant matrix and mean of layers of style image(target) and synthemsized image
            style_targets = [Cov_Mean()(A) for A in self.vgg(style_image, self.style_layers[:])]
            syn_results = [Cov_Mean()(A) for A in self.vgg(syn_image, self.style_layers[:])]

            # pre-process of terms for the evaluation KL divergence:
            # generate new pca covariance matrix and mean for style image and synthesized image with given PCA_basis and reduced ranks.
            PCA_targets = [self.PCA_Proj(data,P ,k) for data,P,k in zip(style_targets,self.PCA_basis,Ks)] 
            PCA_syn_results = [self.PCA_Proj(data,P ,k) for data,P,k in zip(syn_results,self.PCA_basis,Ks)] 
            # evaluate the log term in KL divergence formula
            LogDet_AoverB = [ self.Det(syn[0],tar[0]) for syn,tar,k in zip(PCA_syn_results,PCA_targets,Ks)]

            KLs = []
            # A list of terms needed for KL divergence
            KL_parts  = [ (torch.trace(torch.mm( y[0].inverse(), x[0])), torch.mm( torch.mm((y[1] -x[1]),  y[0].inverse()), (y[1]-x[1]).t() ).squeeze()[0] ,-k, logD) for x,y,logD, k in zip(PCA_syn_results,PCA_targets,LogDet_AoverB,Ks)]

            KLs.append(np.sum(x) for x in KL_parts )  # np.sum(x) gives the 2*KL divergence of each layer
            Es = [ str(-np.log(x)+ np.log(2)) for x in KLs[0]] # E value of each layer is -log(KL)

            # write the reuslts to output file
            new = sp[:3]+Es +["\n"]  
            name = '\t'.join(new) 
            output_list.write(name)

        input_list.close()
        output_list.close()

