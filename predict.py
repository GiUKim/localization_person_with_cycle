import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from glob import glob
from PIL import Image
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
from model import *
from tqdm import tqdm
from torchvision.models import resnet18

def run():
    torch.multiprocessing.freeze_support()
    print('loop')
no_cuda=False
use_cuda = not no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'
print('device:', device)
print('cuda:', use_cuda)
    
config = Config()

def compose_directory():
    if not os.path.exists(config.predict_dst_path):
        os.makedirs(config.predict_dst_path)
    else:
        if config.predict_remove_exist_dir:
            shutil.rmtree(config.predict_dst_path)
            os.makedirs(config.predict_dst_path)
    if not os.path.exists(os.path.join(config.predict_dst_path, 'cycle')):
        os.makedirs(os.path.join(config.predict_dst_path, 'cycle'))
    else:
        if config.predict_remove_exist_dir:
            shutil.rmtree(os.path.join(config.predict_dst_path, 'cycle'))
            os.makedirs(os.path.join(config.predict_dst_path, 'cycle'))
    if not os.path.exists(os.path.join(config.predict_dst_path, 'person_cycle')):
        os.makedirs(os.path.join(config.predict_dst_path, 'person_cycle'))
    else:
        if config.predict_remove_exist_dir:
            shutil.rmtree(os.path.join(config.predict_dst_path, 'person_cycle'))
            os.makedirs(os.path.join(config.predict_dst_path, 'person_cycle'))

if __name__=="__main__":
    run()
    compose_directory()
    model = Net()
    ############
    # ----- resnet 18 use -> size(224, 224, isColor=True 자동 수정), grad-cam(cbr9->layer4 로 쟈동 수정됨) #
    if config.use_res18:
        model = call_resnet18()
    elif config.use_resnext50:
        model = call_resnext50_32x4d()
    ###########
    print(model)
 
    path = config.predict_pretrained_model_path
    verbose_score = config.predict_verbose_score ######## 파일명 뒤에 예측스코어값 붙여서 저장할건지?

    model.load_state_dict(torch.load(path)['model_state_dict'])
    model.eval()
    
    dir = config.predict_src_path
    imgs = glob(dir + '/*.jpg')
    for img in tqdm(imgs):
        if config.isColor:
            image = Image.open(img)  # get image
        else:
            image = Image.open(img).convert('L')  # get image
        i = image.resize((config.width, config.height))
        trans = transforms.ToTensor()
        bi = trans(i)
        bbi = bi.unsqueeze(0)
    
        predict = model(bbi)
        if config.use_res18:
            predict = predict.squeeze()
        elif config.use_resnext50:
            predict = predict.squeeze()
        print('--------------------------')
        print(img.split('/')[-1])
        print('score ->', f'[{round(predict[0].item(), 3)}]')
        print('--------------------------')
        
        score = str(round(predict[0].item(), 4))
        predict_label = predict.data.cpu().numpy().tolist()
        predict_label_line = ' '.join([str(l) for l in predict_label]) + '\n'
        i = cv2.imread(img)
        imh, imw, _ = i.shape
        pred_x1 = int(imw * (predict_label[1] - predict_label[4] / 2.))
        pred_y1 = int(imh * (predict_label[2] - predict_label[3] / 2.))
        pred_x2 = int(imw * (predict_label[1] + predict_label[4] / 2.))
        pred_y2 = int(imh * (predict_label[2] + predict_label[3] / 2.))

        cv2.rectangle(i, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 255, 0), 1)

        if predict_label[0] >= 0.5:
            if verbose_score:
                shutil.copy(img, os.path.join(config.predict_dst_path, 'person_cycle', img.split('/')[-1].split('.jpg')[0]+'_['+score+'].jpg'))
                cv2.imwrite(os.path.join(config.predict_dst_path, 'person_cycle', img.split('/')[-1].split('.jpg')[0]+'_['+score+']_BBOX.jpg'), i)
                f = open(os.path.join(config.predict_dst_path, 'person_cycle', img.split('/')[-1].split('.jpg')[0]+'_['+score+'].txt'), 'w')
                f.write(predict_label_line)
                f.close()

            else:
                shutil.copy(img, os.path.join(config.predict_dst_path, 'person_cycle'))
                cv2.imwrite(os.path.join(config.predict_dst_path, 'person_cycle', img.split('/')[-1].replace('.jpg', '_BBOX.jpg')), i)
                f = open(os.path.join(config.predict_dst_path, 'person_cycle', img.split('/')[-1].replace('.jpg', '.txt')), 'w')
                f.write(predict_label_line)
                f.close()
        else:
            if verbose_score:
                shutil.copy(img, os.path.join(config.predict_dst_path, 'cycle', img.split('/')[-1].split('.jpg')[0]+'_['+score+'].jpg'))
                cv2.imwrite(os.path.join(config.predict_dst_path, 'cycle', img.split('/')[-1].split('.jpg')[0]+'_['+score+']_BBOX.jpg'), i)
                f = open(os.path.join(config.predict_dst_path, 'cycle', img.split('/')[-1].split('.jpg')[0]+'_['+score+'].txt'), 'w')
                f.write(predict_label_line)
                f.close()
            else:
                shutil.copy(img, os.path.join(config.predict_dst_path, 'cycle'))
                cv2.imwrite(os.path.join(config.predict_dst_path, 'cycle', img.split('/')[-1].replace('.jpg', '_BBOX.jpg')), i)
                f = open(os.path.join(config.predict_dst_path, 'cycle', img.split('/')[-1].replace('.jpg', '.txt')), 'w')
                f.write(predict_label_line)
                f.close()


