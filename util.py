from config import Config
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
import random
import torch
from glob import glob
from PIL import Image
import cv2
import numpy as np 
config = Config()

def print_validation_eval_log(correct_list):
    for _idx, cls_correct in enumerate(correct_list):
        if not config.has_unknown:
            if _idx == len(config.class_list):
                break
            print('Class {:>2}: {:<11} Accuracy: {:>8}/{:<8} ({:.2f}%)'.format(
                _idx, config.class_list[_idx], correct_list[_idx], config.test_class_each_num[_idx],
                100. * (correct_list[_idx] / config.test_class_each_num[_idx])
            )
            )
        else:
            if _idx == len(config.class_list) - 1:  # unknown acc
                print('Class {:>2}: {:<11} Accuracy: {:>8}/{:<8} ({:.2f}%)'.format(
                    'N', "unknown", correct_list[-1], config.test_class_each_num[config.unknown_idx],
                    100. * (correct_list[-1] / config.test_class_each_num[config.unknown_idx])
                )
                )
            elif _idx >= config.unknown_idx and config.unknown_idx != len(config.class_list) - 1:
                print('Class {:>2}: {:<11} Accuracy: {:>8}/{:<8} ({:.2f}%)'.format(
                    _idx, config.class_list[_idx + 1], correct_list[_idx], config.test_class_each_num[_idx + 1],
                    100. * (correct_list[_idx] / config.test_class_each_num[_idx + 1])
                )
                )
            else:
                print('Class {:>2}: {:<11} Accuracy: {:>8}/{:<8} ({:.2f}%)'.format(
                    _idx, config.class_list[_idx], cls_correct, config.test_class_each_num[_idx],
                    100. * (cls_correct / config.test_class_each_num[_idx])
                )
                )
    print('=' * 80)

def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
def get_gaussian_filter() :
    kernel = np.array(
                [[0, 0, 0, 0, 0],
                [0, 1.0/25.0, 2.0/25.0, 1.0/25.0, 0],
                [0, 2.0/25.0, 4.0/25.0, 2.0/25.0, 0],
                [0, 1.0/25.0, 2.0/25.0, 1.0/25.0, 0],
                [0, 0, 0, 0, 0]])
    return kernel

def get_laplacian_black_filter() :
    kernel = np.array(
                [[0, 0, 1.0 / 25.0, 0, 0],
                [0, 1.0 / 25.0, 26.0 / 25.0, 1.0 / 25.0, 0],
                [1.0 / 25.0, 26.0 / 25.0, -112.0 / 25.0, 26.0 / 25.0, 1.0 / 25.0],
                [0, 1.0 / 25.0, 26.0 / 25.0, 1.0 / 25.0, 0],
                [0, 0, 1.0 / 25.0, 0, 0]])
    return kernel

def transform_img_to_resize(img, img_width, img_height):
    height, width = img.shape[:2]
    if width > img_width or height > img_height:
        img = cv2.resize(img, (img_width, img_height), cv2.INTER_LINEAR)
    elif width < img_width or height < img_height:
        img = cv2.resize(img, (img_width, img_height), cv2.INTER_LINEAR)
    return img

def change_feature_image(img) :
    img = img * 255
    img = img.astype(np.uint8)
    if config.isColor:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img_denose = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(img_denose, cv2.COLOR_RGB2GRAY)

    height, width = img.shape[:2]
    if config.width != width or config.height != height:
        gray = transform_img_to_resize(gray, config.width, config.height)
    kernel = get_gaussian_filter()
    gray = cv2.filter2D(gray, -1, kernel)
    kernel = get_laplacian_black_filter()
    binary1 = cv2.filter2D(gray, -1, kernel)
    t, t_otsu = cv2.threshold(binary1, -1, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, hierachy = cv2.findContours(t_otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_KCOS)
    img_result = gray.copy().astype('uint8')
    cv2.drawContours(img_result, contours, -1, (0, 0, 0), 1)
    return img_result , t_otsu

def apply_custom_aug(image, **kwargs):
    if config.isColor:
        h, w, _ = image.shape
    else:
        h, w = image.shape
    ret, _ = change_feature_image(image)
    if config.isColor:
        ret = cv2.cvtColor(ret, cv2.COLOR_GRAY2RGB)
    else:
        pass
    ret = cv2.resize(ret, (config.width, config.height))
    ret = ret / 255.
    ret = ret.astype(np.float32)
    return ret

def preprocess_image(img):
    if config.isColor:
        preprocessed_img = img.copy()[:, :, ::-1]
        preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    else:
        preprocessed_img = np.expand_dims(img, -1)
        preprocessed_img = np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input

def threshold(x):
    mean_ = x.mean()
    std_ = x.std()
    thres = mean_ + std_
    x = (x > thres)
    return x

def normalize(Ac):
    Ac_shape = Ac.shape
    AA = Ac.view(Ac.size(0), -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    scaled_ac = AA.view(Ac_shape)
    return scaled_ac

def tensor2image(x, i=0):
    x = normalize(x)
    x = x[i].detach().cpu().numpy()
    x = np.transpose(x, (1, 2, 0))
    return x


def get_display_class_num():
    if len(config.class_list_without_unknown) < 5:
        display_class_num = len(config.class_list_without_unknown)
    else:
        display_class_num = 5 # fix num
    return display_class_num

def visualize_grad_cam(display_class_list, display_class_num, value, model, epoch, accuracy):
    fig = plt.figure(figsize=(18, 9))
    plt.subplots_adjust(bottom=0.01)

    img_path_list = random.sample(glob(config.val_dir + '/*.jpg'), config.visualize_sample_num)
    for sub_image_index, img_path in enumerate(img_path_list):
        cur_display_position = (2 * display_class_num * sub_image_index) + 1
        img_path = random.choice(glob(config.val_dir + '/*.jpg'))
        if config.isColor:
            img = cv2.imread(img_path, 1)
        else:
            img = cv2.imread(img_path, 0)
        if config.isColor:
            img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img_show = img.copy()
        img_show = cv2.resize(img_show, (config.height, config.width))
        img_show2 = img_show.copy()
        img = np.float32(cv2.resize(img, (config.height, config.width))) / 255
        in_tensor = preprocess_image(img).cuda()  # numpy to torch tensor
        output = model(in_tensor)
        
        target_index = 0 # confidence layer 
        target_score = output.data.cpu().numpy()[0]
        target_coord = output.data.cpu().numpy()[1:]
        gt_file = open(img_path.replace('.jpg', '.txt'), 'r')
        line = gt_file.readline()
        gt_file.close()
        gt_score = float(line.split(' ')[0])
        gt_coord = [float(f) for f in line.split(' ')[1:]]
        gt_x1 = int(config.width * (gt_coord[0] - gt_coord[3] / 2.))
        gt_y1 = int(config.height * (gt_coord[1] - gt_coord[2] / 2.))
        gt_x2 = int(config.width * (gt_coord[0] + gt_coord[3] / 2.))
        gt_y2 = int(config.height * (gt_coord[1] + gt_coord[2] / 2.))
        img_show = cv2.rectangle(img_show, (gt_x1, gt_y1), (gt_x2, gt_y2), (0, 0, 255), 1) 
                
        if gt_score < 0.5:
            if target_score < 0.5:
                is_correct = True
            else:
                is_correct = False
        elif gt_score > 0.5:
            if target_score >= 0.5:
                is_correct = True
            else:
                is_correct = False

        if config.use_res18:
            output = output.squeeze()
        elif config.use_resnext50:
            output = output.squeeze()
        output[target_index].sum().backward(retain_graph=True)
        layer4 = value['activations']
        gradient = value['gradients']
        g = torch.mean(gradient, dim=(2,3), keepdim=True)
        grad_cam = layer4 * g
        grad_cam = torch.sum(grad_cam, dim=(0,1))
        grad_cam = torch.clamp(grad_cam, min=0)
        g_2 = gradient**2
        g_3 = gradient**3
        alpha_numer = g_2
        alpha_denom = 2 * g_2 + torch.sum(layer4 * g_3, axis=(2, 3), keepdims=True)
        alpha = alpha_numer / alpha_denom
        w = torch.sum(alpha * torch.clamp(gradient, min=0), axis=(2, 3), keepdims=True)
        grad_cam = grad_cam.data.cpu().numpy()
        grad_cam = cv2.resize(grad_cam, (config.height, config.width))
        if target_score >= 0.5:
            pred_x1 = int(config.width * (target_coord[0] - target_coord[3] / 2.))
            pred_y1 = int(config.height * (target_coord[1] - target_coord[2] / 2.))
            pred_x2 = int(config.width * (target_coord[0] + target_coord[3] / 2.))
            pred_y2 = int(config.height * (target_coord[1] + target_coord[2] / 2.))
            
            if gt_score > 0.5: # correct bbox reg
                img_show = cv2.rectangle(img_show, (pred_x1, pred_y1), (pred_x2, pred_y2), (0, 255, 0), 1)
            else: # wrong bbox reg
                img_show = cv2.rectangle(img_show, (pred_x1, pred_y1), (pred_x2, pred_y2), (255, 0, 0), 1)
        

        plt.subplot(config.visualize_sample_num, 2 * display_class_num, cur_display_position)
        if cur_display_position < 2 * display_class_num and cur_display_position % 2 == 1:
            title_obj = plt.title('image')
            plt.setp(title_obj, color='y')
        if config.isColor:
            plt.imshow(img_show)
        else:
            plt.imshow(img_show, cmap='gray')
        plt.axis('off')
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplot(config.visualize_sample_num, 2 * display_class_num, cur_display_position + 1)
        if config.has_unknown and target_score < config.evaluate_threshold and cur_class_index == target_index:
            title_obj = plt.title("conf (%0.2f)"%(target_score))
        else:
            title_obj = plt.title("conf (%0.2f)"%(target_score))
        
        if is_correct:
            plt.setp(title_obj, color='g')
        else:
            plt.setp(title_obj, color='r')
        plt.subplots_adjust(wspace=0.1, hspace=0)
        plt.imshow(grad_cam, cmap='seismic')
        plt.imshow(img_show2, alpha=.5, cmap='gray')
        plt.axis('off')
    fig.canvas.draw()
    plt.close()
    display_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    display_img = display_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    display_img = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
    display_img = display_img[0:display_img.shape[0], display_img.shape[1]//19:display_img.shape[1]//3]
    cv2.putText(img=display_img, text=f"Epoch: {epoch} Accuracy: {round(accuracy, 3)}%", org=(15, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 220, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.imshow('GRAD-CAM', display_img)
    cv2.waitKey(1)

def visualize(model, epoch, accuracy):
    value = dict()
    def forward_hook(module, input, output):
        value['activations'] = output
    def backward_hook(module, input, output):
        value['gradients'] = output[0]
    target_layer = eval(config.visualize_layer)
    target_layer.register_forward_hook(forward_hook)
    target_layer.register_backward_hook(backward_hook)
    display_class_num = get_display_class_num()
    display_class_list = random.sample(config.class_list_without_unknown, display_class_num)
    visualize_grad_cam(display_class_list, display_class_num, value, model, epoch, accuracy)
    
import torch.nn as nn
import torch.nn.functional as F
import math

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        a_x1 = inputs[..., 0] - inputs[..., 2] / 2.
        a_y1 = inputs[..., 1] - inputs[..., 3] / 2.
        a_x2 = inputs[..., 0] + inputs[..., 2] / 2.
        a_y2 = inputs[..., 1] + inputs[..., 3] / 2.

        b_x1 = targets[..., 0] - targets[..., 2] / 2.
        b_y1 = targets[..., 1] - targets[..., 3] / 2.
        b_x2 = targets[..., 0] + targets[..., 2] / 2.
        b_y2 = targets[..., 1] + targets[..., 3] / 2.

        box1_area = (a_x2 - a_x1) * (a_y2 - a_y1)
        box2_area = (b_x2 - b_x1) * (b_y2 - b_y1)
        
        x1 = torch.max(a_x1, b_x1)
        y1 = torch.max(a_y1, b_y1)
        x2 = torch.min(a_x2, b_x2)
        y2 = torch.min(a_y2, b_y2)
        #x1 = a_x1 if a_x1 > b_x1 else b_x1
        #y1 = a_y1 if a_y1 > b_y1 else b_y1
        #x2 = a_x2 if a_x2 < b_x2 else b_x2
        #y2 = a_y2 if a_y2 < b_y2 else b_y2
        w = torch.clamp(x2 - x1, min=0.0)
        h = torch.clamp(y2 - y1, min=0.0)
        #w = 0 if 0 > x2 - x1 else x2 - x1
        #h = 0 if 0 > y2 - y1 else y2 - y1

        inter = w * h
        iou = inter / (box1_area + box2_area - inter + 1e-7)

        
        #comment out if your model contains a sigmoid or equivalent activation layer
#        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        #inputs = inputs.view(-1)
        #targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        #intersection = (inputs * targets).sum()
        #total = (inputs + targets).sum()
        #union = total - intersection 
        #
        #IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - iou

#def ciou_loss(output, target):
#    eps = 1e-9
#    v = (4. / math.pi) * (torch.arctan(target[:, 3] / (target[:, 2] + eps)) - torch.arctan(output[:, 3] / (output[:, 2] + eps))) ** 2
#    p_square = (target[:, 0] - output[:, 0]) ** 2 + (target[:, 1] - output[: 1]) ** 2
#    c_square = 
