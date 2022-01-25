import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch

import sys
sys.path.append(os.environ['DIR_ROOT'])

from datasets.generic import ImageListRelevants
from utils.pytorch_loader import get_loader
from utils import  common
import nets as nets
from utils.common import tonumpy, matmul, pool
import cv2



# os.environ['DIR_ROOT'] = "work/ImageRetrival/dirtorch"
# os.environ['DB_ROOT'] = "work/ImageRetrival/dirtorch/datasets"

def load_model(path, iscuda):
    checkpoint = common.load_checkpoint(path, iscuda)
    net = nets.create_model(pretrained="", **checkpoint['model_options'])
    net = common.switch_model_to_cuda(net, iscuda, checkpoint)
    net.load_state_dict(checkpoint['state_dict'])
    net.preprocess = checkpoint.get('preprocess', net.preprocess)
    if 'pca' in checkpoint:
        net.pca = checkpoint.get('pca')
    return net

def extractQueryFeature(img,net):
    #print(img.shape)
    net.eval()
    with torch.no_grad():
        query_feature = net(img)
    return query_feature

def resizeImg(img, scale):
    h,w, _ = img.shape
    return cv2.resize(img,(int(w*scale), int(h*scale)))

# def query(img_path, net, topk):
#     query_img = cv2.imread(img_path)
#     cv2_imshow(resizeImg(query_img,0.20))

#     query_feature = extractQueryFeature(query_path,net)
#     query_feature = tonumpy(query_feature)
#     scores = matmul(query_feature, bdescs)

#     ranklist = sorted(range(len(scores)), key=lambda i: scores[i])[-topk:]
#     ranklist = ranklist[::-1] #reverse

#     for i in ranklist:
#         img_path = dataset.get_filename(i)
#         img = cv2.imread(img_path)
#         cv2_imshow(resizeImg(img,0.20))


def resizeImg(img, scale):
    h,w, _ = img.shape
    return cv2.resize(img,(int(w*scale), int(h*scale)))
    


# def query(query_path, net, topk):
#     query_img = cv2.imread(query_path)
#     #cv2_imshow(resizeImg(query_img,0.20))

#     query_feature = extractQueryFeature(query_path,net)
#     query_feature = tonumpy(query_feature)
#     scores = matmul(query_feature, bdescs)

#     ranklist = sorted(range(len(scores)), key=lambda i: scores[i])[-topk:]
#     ranklist = ranklist[::-1] #reverse

#     for i in ranklist:
#         img_path = dataset.get_filename(i)
#         img = cv2.imread(img_path)
#         print(img_path)
#         #cv2_imshow(resizeImg(img,0.20))

# if __name__ == "__main__":

#     #Load Network
#     net = load_model(os.path.join(os.environ['DIR_ROOT'],"checkpoints/Resnet-101-AP-GeM.pt"),False)

#     #Load data
#     dataset = ImageListRelevants(os.environ["DB_ROOT"]+"/oxford5k/gnd_roxford5k.pkl", os.environ["DB_ROOT"] + "/oxford5k", "jpg" )
#     dataLoader = get_loader(dataset, trf_chain="", preprocess=net.preprocess, iscuda=True,
#                             output=['img'], batch_size=1, threads=1, shuffle=False)
#     #Load feature
#     bdescs = np.load(os.path.join(os.environ['DB_ROOT'] + "/oxford5k", 'feats.bdescs.npy'))
    
#     query(os.path.join(os.environ['DIR_ROOT'],"query.jpg"),net,10)
    
    
    
    

