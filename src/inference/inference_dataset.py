import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms,models,utils
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
import sys
import random
import shutil
from model import Detector
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
from datasets import *
from sklearn.metrics import confusion_matrix, roc_auc_score
import pywt
import warnings
import cv2
from skimage import feature
from skimage import filters
warnings.filterwarnings('ignore')

def main(args, model_path, w):
    model=None
    # if w[0]=="r":model=Detector1()
    # if w[0]=="m":model=Detector2()
    model=Detector()
    model=model.to(device)
    cnn_sd=torch.load(model_path, map_location=torch.device('cpu'))["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048,device=device)
    face_detector.eval()

    if args.dataset == 'FFIW':
        video_list,target_list=init_ffiw()
    elif args.dataset == 'FF':
        video_list,target_list=init_ff_t(args.type)
    elif args.dataset == 'DFD':
        video_list,target_list=init_dfd()
    elif args.dataset == 'DFDC':
        video_list,target_list=init_dfdc()
    elif args.dataset == 'DFDCP':
        video_list,target_list=init_dfdcp()
    elif args.dataset == 'CDF':
        video_list,target_list=init_cdf()
    else:
        NotImplementedError

    output_list=[]
    omit_indices = []
    for filename in tqdm(video_list):
        
        # try:
        face_list,idx_list=extract_frames(filename,args.n_frames,face_detector)
        face_list2 = []

        # with torch.no_grad():
        #     img=torch.tensor(face_list).to(device).float()/255
        #     pred=model(img).softmax(1)[:,1]

        with torch.no_grad():
            for f in range(len(face_list)):
                face = face_list[f].astype('float32')/255
                facee = np.transpose(face.copy(), (1,2,0))


                b, g, r = cv2.split(facee)

                cA_r, (cH_r, cV_r, cD_r) = pywt.dwt2(r, 'sym2', mode='reflect')
                cA_g, (cH_g, cV_g, cD_g) = pywt.dwt2(g, 'sym2', mode='reflect')
                cA_b, (cH_b, cV_b, cD_b) = pywt.dwt2(b, 'sym2', mode='reflect')

                cA_r = cv2.resize(cA_r, (380,380), interpolation=cv2.INTER_LINEAR).astype('float32')
                cA_g = cv2.resize(cA_g, (380,380), interpolation=cv2.INTER_LINEAR).astype('float32')
                cA_b = cv2.resize(cA_b, (380,380), interpolation=cv2.INTER_LINEAR).astype('float32')

                cA_r = (cA_r + r)/2
                cA_g = (cA_g + g)/2
                cA_b = (cA_b + b)/2

                img_dwt = np.array([cA_r, cA_g, cA_b])

                face_list[f] = img_dwt
                # face_list2.append(img_dwt)
            
            img = torch.tensor(face_list).to(device)
            
            # img=torch.tensor(face_list).to(device).float()/255
            # img2 = torch.tensor(face_list2).to(device)

            pred=model(img).softmax(1)[:,1]

            # pred1=model(img).softmax(1)[:,1]
            # pred2=model(img2).softmax(1)[:,1]
            # pred = (pred1+pred2)/2

        # with torch.no_grad():
        #     for f in range(len(face_list)):
        #         face = face_list[f].astype('float32')/255
        #         face = np.transpose(face, (1, 2, 0))

        #         face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        #         cA, _ = pywt.dwt2(face.copy(), 'sym2', mode='symmetric')
        #         cA_resized = cv2.resize(cA, (380,380), interpolation=cv2.INTER_LINEAR).astype('float32')

        #         img = [cA_resized, cA_resized, cA_resized]
        #         img = np.array(img)
        #         face_list[f] = img

        #     img=torch.tensor(face_list).to(device)
        #     pred=model(img).softmax(1)[:,1]
            
            
        pred_list=[]
        idx_img=-1
        for i in range(len(pred)):
            if idx_list[i]!=idx_img:
                pred_list.append([])
                idx_img=idx_list[i]
            pred_list[-1].append(pred[i].item())
        pred_res=np.zeros(len(pred_list))
        for i in range(len(pred_res)):
            pred_res[i]=max(pred_list[i])
        pred=pred_res.mean()

        output_list.append(pred)
        # except Exception as e:
        #     # pred=0.5
        #     print(e)
        #     continue
        # output_list.append(pred)

    auc=roc_auc_score(target_list,output_list)
    print(f'{args.dataset}| AUC: {auc:.4f}')


if __name__=='__main__':
    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('mps')

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-d',dest='dataset',type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    parser.add_argument('-t',dest='type',default="Face2Face",type=str)
    args=parser.parse_args()

    # weights = ['14_0.9988_val.tar']
    # weights = ['e_sbi380_e85_v9990.tar', 'e_sbi380_e89_9988.tar','e_sbi48_e4_v9994.tar','e_sbi48_e16_v9990.tar','e_sbi48_e20_v9991.tar','e_sbi48_e22_v9991.tar','e_sbi48_e25_v9988.tar','sbi380_e88_v9991.tar','sbi380_e91_v9992.tar', 'eb5_RBG.tar', '4_0.9994_val.tar',]
    weights = os.listdir("./weights")
    
    for w in weights:
        if w[0] == "d":continue
        print(w)
        main(args, os.path.join("./weights",w),w)
        print("-------------------------------------------------------------------")

