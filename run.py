from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import random
import torch
import os
import sys
sys.path.append('align_methods')
from align import align, re_align
from utils import save_priv, cosdistance, inference
import math
import argparse
from scipy.misc import imread
from collections import OrderedDict
from get_model import getmodel
from tqdm import tqdm
from input_diversify import input_diversity
import cv2
from config import threshold
from torch.autograd import Variable
from mmd import mmd_loss

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', help='Input directory with images.', type=str, default='data/pairs.txt')
parser.add_argument('--num_iter', help='Number of iterations.', type=int, default=100)
parser.add_argument('--src_model', help='White-box model', type=str, default='ArcFace', choices=threshold.keys())
parser.add_argument('--ensemble_image', help='whether to use multiple images', type=int, default=0, choices=[0, 1])
parser.add_argument('--batch_size', help='batch size', type=int, default=1)
parser.add_argument('--output', help='output directory', type=str, default='output/exppriv')
parser.add_argument('--gain', help='gain function', type=str, default='gain3')
parser.add_argument('--target_nums', help='target_nums', type=int, default=10)
parser.add_argument('--gamma', help='gamma', type=float, default=0.0)
parser.add_argument('--norm', help='select norm', type=str, default='l2')
args = parser.parse_args()

seed = 0


def main():
    model, img_shape = getmodel(args.src_model)
    iters = args.num_iter

    cnt_pairs = 0
    candidate_e = [12]
    func = {
        'gain1': Gain1,
        'gain2': Gain2,
        'gain3': Gain3
        }
    
    # conduct attacker
    attacker_persons = []
    with open(args.input_file, 'r') as f:
        for pair in f.readlines()[:500]:
            attacker_person = pair.strip()
            attacker_persons.append(attacker_person)
            
    random.seed(seed)
    cnt = 0
    batch_size = args.batch_size

    # conduct targets, only 10 identities!
    target_nums = args.target_nums
    aligned_target_imgs = []
    target_persons = []
    with open('targets/target_attacks.txt', 'r') as f:
        for line in f.readlines():
            target_person = line.strip()
            target_persons.append(target_person)

    for i in range(target_nums):
        target_img = imread(target_persons[i]).astype(np.float32)
        align_target_img, _= align(target_img)
        align_target_img = cv2.resize(align_target_img, (img_shape[1], img_shape[0]))
        aligned_target_imgs.append(align_target_img)
    aligned_target_imgs = np.array(aligned_target_imgs)        

    # extract features of targets
    target_feas = inference(aligned_target_imgs, model, image_shape = img_shape)
    target_feas = Variable(target_feas)
    
    # extract initial features of attackers
    n = len(attacker_persons)
    m = 1.0
    for i in tqdm(range(0, n, batch_size)):
        l, r = i, min(i + batch_size, n)
        original_images = []
        aligned_images = []
        aligned_names = []
        M = []
        for j in range(l, r):
            attack_name, id1, id2 = attacker_persons[j].split('\t')
            original_img = imread("data/lfw/{}/{}_{:04d}.jpg".format(attack_name, attack_name, int(id2))).astype(np.float32)
            original_images.append(original_img)
            align_img, tmpM = align(original_img)
            align_img = cv2.resize(align_img, (img_shape[1], img_shape[0]))
            aligned_images.append(align_img)
            aligned_names.append("{}_{:04d}".format(attack_name, int(id2)))
            M.append(tmpM)
        aligned_images = np.array(aligned_images)
        input_init = torch.Tensor(aligned_images).cuda()
        input_init = input_init.permute(0, 3, 1, 2)
        init_feas = model.forward(input_init)
        init_feas = Variable(init_feas)

        # craft protected images 
        for epsilon in candidate_e:
            alpha = 1.5 * epsilon / iters
            inputs = torch.Tensor(aligned_images.copy()).cuda()
            inputs = inputs.permute(0, 3, 1, 2)
            sum_grad = torch.zeros_like(inputs)
            min_img = torch.clamp(inputs - epsilon, min=0)
            max_img = torch.clamp(inputs + epsilon, max=255)
            adv_images = inputs.detach().clone().requires_grad_(True)
            for iii in range(iters):
                std_proj = random.uniform(0.01, 0.1)
                std_rotate = random.uniform(0.01, 0.1)
                tmp_advs = []
                tmp_grads = []
                tmp_losses = []
                model.zero_grad()
                # introduce input diversity for generalizaion
                images = input_diversity(adv_images, std_proj, std_rotate)
                adv_feas = model.forward(images)

                # compute image-level natural loss by MMD. 
                loss_mmd = mmd_loss(adv_images.clone().reshape(batch_size,-1), 
                        inputs.clone().reshape(batch_size,-1))
                
                # search optimal target 
                for idx in range(target_nums):
                    model.zero_grad()
                    loss_i = torch.mean((adv_feas - init_feas) ** 2)
                    loss_t = torch.mean((adv_feas - target_feas[idx]) ** 2)

                    # total loss 
                    '''
                    loss_mmd does not affect search direction of targets set, just for updating images.
                    '''
                    loss = loss_t - loss_i + args.gamma * loss_mmd
   
                    loss.backward(retain_graph = True)
                    grad = adv_images.grad.data.clone()
                    grad = grad / grad.abs().mean(dim=[1, 2, 3], keepdim=True)
                    
                    tmp_sum_grad = m * sum_grad.clone() + grad
                    adv_images.grad.data.zero_()
                    tmp_adv_images = adv_images.data.clone()
                    # infty norm
                    if args.norm == 'linf':
                        tmp_adv_images = tmp_adv_images - torch.sign(tmp_sum_grad) * alpha
                    elif args.norm == 'l2':
                        factor = np.sqrt(np.prod(img_shape)* 3)
                        grad2d = tmp_sum_grad.reshape((tmp_sum_grad.size(0), -1))
                        gradnorm = grad2d.norm(p=2, dim=1, keepdim=True)
                        grad_unit = grad2d / gradnorm
                        delta = -torch.reshape(grad_unit, tmp_sum_grad.size()) * alpha * factor
                        tmp_adv_images = tmp_adv_images + delta
                    
                    tmp_adv_images = torch.max(tmp_adv_images, min_img)
                    tmp_adv_images = torch.min(tmp_adv_images, max_img)
                    tmp_grads.append(tmp_sum_grad)
                    tmp_advs.append(tmp_adv_images)
                    tmp_losses.append(loss.data.unsqueeze(0))
               
                tmp_losses = torch.cat(tmp_losses)
                # find best index
                best_idx = submodular(target_feas, init_feas, tmp_advs, model, func[args.gain])
                sum_grad = tmp_grads[best_idx]
                
                adv_images = tmp_advs[best_idx]
                print(tmp_losses[best_idx], best_idx)
                adv_images = adv_images.detach().requires_grad_(True)
            adv_images = adv_images.detach().permute(0, 2, 3, 1).cpu().numpy()
                
            for j in range(batch_size):
                attacker_name = attacker_persons[l + j].split('\t')[0]
                _ = save_priv(adv_images[j], aligned_images[j], original_images[j], attacker_name, M[j], epsilon, args.src_model, args.output)

def submodular(target_feas, init_feas, tmp_advs, model, Gain):
    tmp_advs = torch.cat(tmp_advs)
    
    gains = torch.zeros(size = (len(target_feas),), dtype = torch.float32)
    tmpadv_feas = model.forward(tmp_advs)
    for idx in range(len(tmpadv_feas)):
        gains[idx] = Gain(tmpadv_feas[idx].unsqueeze(0), init_feas, target_feas, idx)
    best_idx = torch.argmax(gains)
    return best_idx

def L2distance(x, y):
	return torch.sqrt(torch.sum((x - y)**2, dim = 1))

def Gain1(adv_fea, init_feas, target_feas, idx=0):
    d1 = L2distance(adv_fea, init_feas)
    d2 = torch.sum(torch.exp(d1 - L2distance(adv_fea, target_feas)))
    return torch.log(1.0 + d2)

def Gain2(adv_fea, init_feas, target_feas, idx=0):
    d1 = L2distance(adv_fea, init_feas)
    d2 = torch.min(torch.exp(d1 - L2distance(adv_fea, target_feas)))
    return torch.log(1.0 + d2)

def Gain3(adv_fea, init_feas, target_feas, idx=0):
    d1 = L2distance(adv_fea, init_feas)
    d2 = torch.max(torch.exp(d1 - L2distance(adv_fea, target_feas)))
    return torch.log(1.0 + d2)


if __name__ == '__main__':
    main()
