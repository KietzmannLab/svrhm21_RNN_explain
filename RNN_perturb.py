# Script to perform perturbation analyses on the trained RNN
# Requires tensorflow 1.13, python 3.7, scikit-learn, and pytorch 1.6.0

############################# IMPORTING MODULES ##################################

import torch
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import zoom
from scipy.ndimage import rotate
from random import shuffle

############################# GET DATASETS ##################################

class MNISTData:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

class MNIST:
    def __init__(self, dataset_name='mnist'):
        if dataset_name == 'mnist':
            dataset = datasets.MNIST
        elif dataset_name == 'fashionmnist':
            dataset = datasets.FashionMNIST
        else:
            raise ValueError("dataset_name must be either 'mnist' or 'fashionmnist'")
        
        # Load datasets
        train_dataset = dataset(root='./data', train=True, download=True)
        test_dataset = dataset(root='./data', train=False, download=True)
        
        # Extract images and labels
        train_images = train_dataset.data.numpy()
        train_labels = self.one_hot(train_dataset.targets.numpy())
        test_images = test_dataset.data.numpy()
        test_labels = self.one_hot(test_dataset.targets.numpy())
        
        # Create validation set from the training set
        val_size = 5000
        train_size = len(train_images) - val_size
        
        train_images, val_images = train_images[:train_size], train_images[train_size:]
        train_labels, val_labels = train_labels[:train_size], train_labels[train_size:]
        
        # Store datasets in MNIST-like structure
        self.train = MNISTData(train_images, train_labels)
        self.validation = MNISTData(val_images, val_labels)
        self.test = MNISTData(test_images, test_labels)
        
    def one_hot(self, labels, num_classes=10):
        return np.eye(num_classes)[labels]

# Example usage:
mnist = MNIST(dataset_name='mnist')
fmnist = MNIST(dataset_name='fashionmnist')

############################# FUNCTIONS DEFINED ##################################

# A function to scramble image chunks
def im_scram(im,parts_h): # scramble parts_h*parts_h equal parts of the given image
    win_prop = parts_h
    dimsh = np.shape(im)
    im_new = np.zeros(dimsh)
    dimsh_win = np.floor(dimsh[0]/win_prop)
    n_cells = np.square(np.int(dimsh[0]/dimsh_win))
    cell_c = np.int(dimsh[0]/dimsh_win)
    ind_new = np.linspace(0,n_cells-1,n_cells).astype('int32')
    while np.mean(ind_new == np.linspace(0,n_cells-1,n_cells).astype('int32')) == 1:
        shuffle(ind_new)
    for i in range(n_cells):
        j = ind_new[i]
        im_new[np.int(np.mod(i,cell_c)*dimsh_win):np.int(np.mod(i,cell_c)*dimsh_win+dimsh_win),
               np.int(np.floor(i*1./cell_c*1.)*dimsh_win):np.int(np.floor(i*1./cell_c*1.)*dimsh_win+dimsh_win)] = im[
            np.int(np.mod(j,cell_c)*dimsh_win):np.int(np.mod(j,cell_c)*dimsh_win+dimsh_win),
            np.int(np.floor(j*1./cell_c*1.)*dimsh_win):np.int(np.floor(j*1./cell_c*1.)*dimsh_win+dimsh_win)]
    return im_new

# A function to generate images and the perturbed versions for analysis
def gen_images(n_imgs,n_set): # n_imgsx6 required, set used (0 train, 1 val, 2 test) 8 objects in image (1 is intact), 2 levels of zoom, rotation and x/y pos for each object
    imgs_h = np.zeros([n_imgs,1,100,100])
    imgs_h_xswap = np.zeros([n_imgs,1,100,100])
    imgs_h_yswap = np.zeros([n_imgs,1,100,100])
    imgs_h_rotswap = np.zeros([n_imgs,1,100,100])
    imgs_h_sizeswap = np.zeros([n_imgs,1,100,100])
    imgs_h_catswap_w = np.zeros([n_imgs,1,100,100])
    imgs_h_catswap_b = np.zeros([n_imgs,1,100,100])
    labs_h = np.zeros([n_imgs,20])
    pos_x_h = np.zeros([n_imgs,2])
    pos_y_h = np.zeros([n_imgs,2])
    size_h = np.zeros([n_imgs,2])
    rot_h = np.zeros([n_imgs,2])
    n_objs = 8
    for n_im in np.arange(n_imgs):
        inst_img = np.zeros([100,100])
        inst_img_xswap = np.zeros([100,100])
        inst_img_yswap = np.zeros([100,100])
        inst_img_rotswap = np.zeros([100,100])
        inst_img_sizeswap = np.zeros([100,100])
        inst_img_catswap_w = np.zeros([100,100])
        inst_img_catswap_b = np.zeros([100,100])
        obj_ord = np.linspace(0,n_objs-1,n_objs)
        dum_obj_ind = 4+np.random.randint(n_objs/2)
        dum_obj_ind_xswap = np.int(5.5 + np.sign(dum_obj_ind - 5.5)*(2 - np.abs(dum_obj_ind - 5.5)))
        dum_obj_ind_yswap = np.int(2*(5-np.floor(dum_obj_ind/2))+np.mod(dum_obj_ind,2))
        dum_dat_ord = (np.random.random(8) < 0.5)*1.
        for i in np.arange(n_objs):
            if dum_dat_ord[i] == 0: # dataset M or F
                if n_set == 0:
                    dathh = mnist.train
                elif n_set == 1:
                    dathh = mnist.validation
                elif n_set == 2:
                    dathh = mnist.test
                inst_obj_ind = np.random.randint(np.shape(dathh.images)[0])
                if i == dum_obj_ind:
                    inst_lab = np.where(dathh.labels[inst_obj_ind,:]==1)[0][0]
                inst_obj = np.reshape(dathh.images[inst_obj_ind,:],(28,28))
            else:
                if n_set == 0:
                    dathh = fmnist.train
                elif n_set == 1:
                    dathh = fmnist.validation
                elif n_set == 2:
                    dathh = fmnist.test
                inst_obj_ind = np.random.randint(np.shape(dathh.images)[0])
                if i == dum_obj_ind:
                    inst_lab = 10 + np.where(dathh.labels[inst_obj_ind,:]==1)[0][0]
                inst_obj = np.reshape(dathh.images[inst_obj_ind,:],(28,28))
            if i == dum_obj_ind:
                if dum_dat_ord[i] == 0: # dataset M or F
                    if n_set == 0:
                        dathh = mnist.train
                    elif n_set == 1:
                        dathh = mnist.validation
                    elif n_set == 2:
                        dathh = mnist.test
                    inst_obj_ind_catswap_w = np.random.randint(np.shape(dathh.images)[0])
                    while np.where(dathh.labels[inst_obj_ind_catswap_w,:]==1)[0][0] == inst_lab:
                        inst_obj_ind_catswap_w = np.random.randint(np.shape(dathh.images)[0])
                    inst_obj_catswap_w = np.reshape(dathh.images[inst_obj_ind_catswap_w,:],(28,28))
                    if n_set == 0:
                        dathh = fmnist.train
                    elif n_set == 1:
                        dathh = fmnist.validation
                    elif n_set == 2:
                        dathh = fmnist.test
                    inst_obj_ind_catswap_b = np.random.randint(np.shape(dathh.images)[0])
                    inst_obj_catswap_b = np.reshape(dathh.images[inst_obj_ind_catswap_b,:],(28,28))
                else:
                    if n_set == 0:
                        dathh = fmnist.train
                    elif n_set == 1:
                        dathh = fmnist.validation
                    elif n_set == 2:
                        dathh = fmnist.test
                    inst_obj_ind_catswap_w = np.random.randint(np.shape(dathh.images)[0])
                    while np.where(dathh.labels[inst_obj_ind_catswap_w,:]==1)[0][0] == inst_lab:
                        inst_obj_ind_catswap_w = np.random.randint(np.shape(dathh.images)[0])
                    inst_obj_catswap_w = np.reshape(dathh.images[inst_obj_ind_catswap_w,:],(28,28))
                    if n_set == 0:
                        dathh = mnist.train
                    elif n_set == 1:
                        dathh = mnist.validation
                    elif n_set == 2:
                        dathh = mnist.test
                    inst_obj_ind_catswap_b = np.random.randint(np.shape(dathh.images)[0])
                    inst_obj_catswap_b = np.reshape(dathh.images[inst_obj_ind_catswap_b,:],(28,28))
            dumh111 = (np.random.random(1)[0] > 0.5)*1
            if dumh111 == 0: # zoom 0.9 or 1.5
                inst_obj = zoom(inst_obj,0.8)
                if i == dum_obj_ind:
                    inst_obj_sizeswap = zoom(inst_obj,1.6)
                    inst_obj_catswap_w = zoom(inst_obj_catswap_w,0.8)
                    inst_obj_catswap_b = zoom(inst_obj_catswap_b,0.8)
            else:
                inst_obj = zoom(inst_obj,1.6)
                if i == dum_obj_ind:
                    inst_obj_sizeswap = zoom(inst_obj,0.8)
                    inst_obj_catswap_w = zoom(inst_obj_catswap_w,1.6)
                    inst_obj_catswap_b = zoom(inst_obj_catswap_b,1.6)
            if i == dum_obj_ind:
                size_h[n_im,dumh111] = 1.
            dumh111 = (np.random.random(1)[0] > 0.5)*1
            if dumh111 == 0: # rotate 30 or -30
                dumrot = 35
                inst_obj = rotate(inst_obj,dumrot,reshape=False)
                if i == dum_obj_ind:
                    inst_obj_rotswap = rotate(inst_obj,-35,reshape=False)
                    inst_obj_sizeswap = rotate(inst_obj_sizeswap,dumrot,reshape=False)
                    inst_obj_catswap_w = rotate(inst_obj_catswap_w,dumrot,reshape=False)
                    inst_obj_catswap_b = rotate(inst_obj_catswap_b,dumrot,reshape=False)
            else:
                dumrot = -35
                inst_obj = rotate(inst_obj,dumrot,reshape=False) # rotate -25 to -35
                if i == dum_obj_ind:
                    inst_obj_rotswap = rotate(inst_obj,35,reshape=False)
                    inst_obj_sizeswap = rotate(inst_obj_sizeswap,dumrot,reshape=False)
                    inst_obj_catswap_w = rotate(inst_obj_catswap_w,dumrot,reshape=False)
                    inst_obj_catswap_b = rotate(inst_obj_catswap_b,dumrot,reshape=False)
            if i == dum_obj_ind:
                rot_h[n_im,dumh111] = 1.
            if i != dum_obj_ind:
                inst_obj = im_scram(inst_obj,3) # scrambled if not object of interest
            if np.mod(obj_ord[i],4) == 0: # x_loc up or down
                x_loc = np.int(np.round(25))
                y_loc = np.int(np.round(25))
                x_loc_xswap = np.int(np.round(75))
                y_loc_yswap = np.int(np.round(75))
                if i == dum_obj_ind:
                    pos_y_h[n_im,0] = 1.
                    pos_x_h[n_im,0] = 1.
            elif np.mod(obj_ord[i],4) == 1:
                x_loc = np.int(np.round(75)) # 75 +- 2.5
                y_loc = np.int(np.round(25)) # 25 +- 2.5
                x_loc_xswap = np.int(np.round(25))
                y_loc_yswap = np.int(np.round(75))
                if i == dum_obj_ind:
                    pos_y_h[n_im,1] = 1.
                    pos_x_h[n_im,0] = 1.
            elif np.mod(obj_ord[i],4) == 2:
                x_loc = np.int(np.round(25)) # 25 +- 2.5
                y_loc = np.int(np.round(75)) # 75 +- 2.5
                x_loc_xswap = np.int(np.round(75))
                y_loc_yswap = np.int(np.round(25))
                if i == dum_obj_ind:
                    pos_y_h[n_im,0] = 1.
                    pos_x_h[n_im,1] = 1.
            elif np.mod(obj_ord[i],4) == 3:
                x_loc = np.int(np.round(75)) # 75 +- 2.5
                y_loc = np.int(np.round(75)) # 75 +- 2.5
                x_loc_xswap = np.int(np.round(25))
                y_loc_yswap = np.int(np.round(25))
                if i == dum_obj_ind:
                    pos_y_h[n_im,1] = 1.
                    pos_x_h[n_im,1] = 1.
            inst_obj = (inst_obj-np.min(inst_obj))/(np.max(inst_obj)-np.min(inst_obj))
            if i == dum_obj_ind:
                inst_obj_rotswap = (inst_obj_rotswap-np.min(inst_obj_rotswap))/(np.max(inst_obj_rotswap)-np.min(inst_obj_rotswap))
                inst_obj_sizeswap = (inst_obj_sizeswap-np.min(inst_obj_sizeswap))/(np.max(inst_obj_sizeswap)-np.min(inst_obj_sizeswap))
                inst_obj_catswap_w = (inst_obj_catswap_w-np.min(inst_obj_catswap_w))/(np.max(inst_obj_catswap_w)-np.min(inst_obj_catswap_w))
                inst_obj_catswap_b = (inst_obj_catswap_b-np.min(inst_obj_catswap_b))/(np.max(inst_obj_catswap_b)-np.min(inst_obj_catswap_b))
            # print(np.int(np.floor(np.shape(inst_obj)[0]/2)),np.int(np.ceil(np.shape(inst_obj)[0]/2)),np.shape(inst_obj)[0])
            inst_img[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
            if i != dum_obj_ind and i != dum_obj_ind_xswap and i != dum_obj_ind_yswap:
                inst_img_xswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img_xswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
                inst_img_yswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img_yswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
                inst_img_rotswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img_rotswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
                inst_img_sizeswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img_sizeswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
                inst_img_catswap_w[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img_catswap_w[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
                inst_img_catswap_b[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img_catswap_b[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
            if i == dum_obj_ind_xswap:
                inst_img_xswap[x_loc_xswap-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc_xswap+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img_xswap[x_loc_xswap-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc_xswap+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
                inst_img_yswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img_yswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
                inst_img_rotswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img_rotswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
                inst_img_sizeswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img_sizeswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
                inst_img_catswap_w[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img_catswap_w[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
                inst_img_catswap_b[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img_catswap_b[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
            if i == dum_obj_ind_yswap:
                inst_img_xswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img_xswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
                inst_img_yswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc_yswap-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc_yswap+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img_yswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc_yswap-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc_yswap+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
                inst_img_rotswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img_rotswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
                inst_img_sizeswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img_sizeswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
                inst_img_catswap_w[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img_catswap_w[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
                inst_img_catswap_b[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img_catswap_b[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
            if i == dum_obj_ind:
                inst_img_xswap[x_loc_xswap-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc_xswap+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img_xswap[x_loc_xswap-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc_xswap+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
                inst_img_yswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc_yswap-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc_yswap+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img_yswap[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc_yswap-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc_yswap+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
                inst_img_rotswap[x_loc-np.int(np.floor(np.shape(inst_obj_rotswap)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj_rotswap)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj_rotswap)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj_rotswap)[1]/2.))] = (1-inst_obj_rotswap)*inst_img_rotswap[x_loc-np.int(np.floor(np.shape(inst_obj_rotswap)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj_rotswap)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj_rotswap)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj_rotswap)[1]/2.))] + (inst_obj_rotswap)*inst_obj_rotswap
                inst_img_sizeswap[x_loc-np.int(np.floor(np.shape(inst_obj_sizeswap)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj_sizeswap)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj_sizeswap)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj_sizeswap)[1]/2.))] = (1-inst_obj_sizeswap)*inst_img_sizeswap[x_loc-np.int(np.floor(np.shape(inst_obj_sizeswap)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj_sizeswap)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj_sizeswap)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj_sizeswap)[1]/2.))] + (inst_obj_sizeswap)*inst_obj_sizeswap
                inst_img_catswap_w[x_loc-np.int(np.floor(np.shape(inst_obj_catswap_w)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj_catswap_w)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj_catswap_w)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj_catswap_w)[1]/2.))] = (1-inst_obj_catswap_w)*inst_img_catswap_w[x_loc-np.int(np.floor(np.shape(inst_obj_catswap_w)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj_catswap_w)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj_catswap_w)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj_catswap_w)[1]/2.))] + (inst_obj_catswap_w)*inst_obj_catswap_w
                inst_img_catswap_b[x_loc-np.int(np.floor(np.shape(inst_obj_catswap_b)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj_catswap_b)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj_catswap_b)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj_catswap_b)[1]/2.))] = (1-inst_obj_catswap_b)*inst_img_catswap_b[x_loc-np.int(np.floor(np.shape(inst_obj_catswap_b)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj_catswap_b)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj_catswap_b)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj_catswap_b)[1]/2.))] + (inst_obj_catswap_b)*inst_obj_catswap_b
        inst_img = (inst_img-np.min(inst_img))/(np.max(inst_img)-np.min(inst_img))
        inst_img_xswap = (inst_img_xswap-np.min(inst_img_xswap))/(np.max(inst_img_xswap)-np.min(inst_img_xswap))
        inst_img_yswap = (inst_img_yswap-np.min(inst_img_yswap))/(np.max(inst_img_yswap)-np.min(inst_img_yswap))
        inst_img_rotswap = (inst_img_rotswap-np.min(inst_img_rotswap))/(np.max(inst_img_rotswap)-np.min(inst_img_rotswap))
        inst_img_sizeswap = (inst_img_sizeswap-np.min(inst_img_sizeswap))/(np.max(inst_img_sizeswap)-np.min(inst_img_sizeswap))
        inst_img_catswap_w = (inst_img_catswap_w-np.min(inst_img_catswap_w))/(np.max(inst_img_catswap_w)-np.min(inst_img_catswap_w))
        inst_img_catswap_b = (inst_img_catswap_b-np.min(inst_img_catswap_b))/(np.max(inst_img_catswap_b)-np.min(inst_img_catswap_b))
        if np.isnan(np.min(inst_img)) or np.isnan(np.min(inst_img_xswap)) or np.isnan(np.min(inst_img_yswap)) or np.isnan(np.min(inst_img_rotswap)) or np.isnan(np.min(inst_img_sizeswap)) or np.isnan(np.min(inst_img_catswap_w)) or np.isnan(np.min(inst_img_catswap_b)):
            print('NaN in input')
            exit(1)
        imgs_h[n_im,0,:,:] = inst_img
        imgs_h_xswap[n_im,0,:,:] = inst_img_xswap
        imgs_h_yswap[n_im,0,:,:] = inst_img_yswap
        imgs_h_rotswap[n_im,0,:,:] = inst_img_rotswap
        imgs_h_sizeswap[n_im,0,:,:] = inst_img_sizeswap
        imgs_h_catswap_w[n_im,0,:,:] = inst_img_catswap_w
        imgs_h_catswap_b[n_im,0,:,:] = inst_img_catswap_b
        labs_h[n_im,inst_lab] = 1.
    return imgs_h,imgs_h_xswap,imgs_h_yswap,imgs_h_rotswap,imgs_h_sizeswap,imgs_h_catswap_w,imgs_h_catswap_b,labs_h,pos_x_h,pos_y_h,size_h,rot_h

# Defining the RNN class for extracting representations and original recurrent flows
class RNNet_all_fbr(nn.Module):
    def __init__(self, n_feats=8, ker_size=5,t_steps=3,b_flag=0,g_flag=1,l_flag=1,t_flag=1):
        super(RNNet_all_fbr, self).__init__()
        self.conv1 = nn.Conv2d(1, n_feats, ker_size)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(n_feats, n_feats*2, ker_size)
        self.fc1 = nn.Linear(n_feats*2 * 9 * 9, n_feats*16)
        self.fc2 = nn.Linear(n_feats*16*t_steps, 20)
        self.dropout = nn.Dropout(0.5)
        self.c1xb = nn.ConvTranspose2d(n_feats,1,7,3) # in_channel, out_channel, kernel_size, stride, padding
        self.c2xb = nn.ConvTranspose2d(n_feats*2,1,20,10)
        self.fc1xb = nn.Linear(n_feats*16, 100*100)
        self.c1c1b = nn.Conv2d(n_feats, n_feats, ker_size, 1, 2)
        self.c2c1b = nn.ConvTranspose2d(n_feats*2,n_feats,16,10)
        self.fc1c1b = nn.Linear(n_feats*16, 96*96*n_feats)
        self.c2c2b = nn.Conv2d(n_feats*2, n_feats*2, ker_size, 1, 2)
        self.fc1c2b = nn.Linear(n_feats*16, 28*28*n_feats*2)
        self.fc1fc1b = nn.Linear(n_feats*16, n_feats*16)
        self.c1xg = nn.ConvTranspose2d(n_feats,1,7,3) # in_channel, out_channel, kernel_size, stride, padding
        self.c2xg = nn.ConvTranspose2d(n_feats*2,1,20,10)
        self.fc1xg = nn.Linear(n_feats*16, 100*100)
        self.c1c1g = nn.Conv2d(n_feats, n_feats, ker_size, 1, 2)
        self.c2c1g = nn.ConvTranspose2d(n_feats*2,n_feats,16,10)
        self.fc1c1g = nn.Linear(n_feats*16, 96*96*n_feats)
        self.c2c2g = nn.Conv2d(n_feats*2, n_feats*2, ker_size, 1, 2)
        self.fc1c2g = nn.Linear(n_feats*16, 28*28*n_feats*2)
        self.fc1fc1g = nn.Linear(n_feats*16, n_feats*16)
        self.n_feats = n_feats
        self.t_steps = t_steps
        self.b_flag = b_flag
        self.g_flag = g_flag
        self.l_flag = l_flag
        self.t_flag = t_flag
    def forward(self, x):
        actvs = {}
        actvs[0] = {}
        actvs[1] = {}
        actvs[2] = {}
        actvs[3] = {}
        fb_acts = {}
        fb_acts[0] = {}
        fb_acts[1] = {}
        fb_acts[2] = {}
        fb_acts[3] = {}
        fb_acts_comb = {}
        fb_acts_comb[0] = {}
        fb_acts_comb[1] = {}
        fb_acts_comb[2] = {}
        fb_acts_comb[3] = {}
        for i in np.arange(2):
            fb_acts[0][i] = {}
            fb_acts[1][i] = {}
            fb_acts[2][i] = {}
            fb_acts[3][i] = {}
            fb_acts_comb[0][i] = {}
            fb_acts_comb[1][i] = {}
            fb_acts_comb[2][i] = {}
            fb_acts_comb[3][i] = {}
            for j in np.arange(3):
                fb_acts[0][i][j] = {}
                fb_acts[1][i][j] = {}
                if j > 0:
                    fb_acts[2][i][j-1] = {}
                    if j > 1:
                        fb_acts[3][i][j-2] = {}
        actvs[0][0] = F.relu(x) - F.relu(x-1)
        c1 = F.relu(self.conv1(actvs[0][0]))
        actvs[1][0] = self.pool(c1)
        c2 = F.relu(self.conv2(actvs[1][0]))
        actvs[2][0] = self.pool(c2)
        actvs[3][0] = F.relu(self.fc1(actvs[2][0].view(-1, self.n_feats*2 * 9 * 9)))
        actvs[4] = actvs[3][0]
        if self.t_steps > 0:
            for t in np.arange(self.t_steps-1):
                fb_acts[0][0][0][t] = self.t_flag*self.c1xb(actvs[1][t])
                fb_acts[0][0][1][t] = self.t_flag*self.c2xb(actvs[2][t])
                fb_acts[0][0][2][t] = self.t_flag*(self.fc1xb(actvs[3][t])).view(-1,1,100,100)
                fb_acts_comb[0][0][t] = self.b_flag*(fb_acts[0][0][0][t] + fb_acts[0][0][1][t] + fb_acts[0][0][2][t])
                fb_acts[0][1][0][t] = self.t_flag*self.c1xg(actvs[1][t])
                fb_acts[0][1][1][t] = self.t_flag*self.c2xg(actvs[2][t])
                fb_acts[0][1][2][t] = self.t_flag*(self.fc1xg(actvs[3][t])).view(-1,1,100,100)
                fb_acts_comb[0][1][t] = self.g_flag*(fb_acts[0][1][0][t] + fb_acts[0][1][1][t] + fb_acts[0][1][2][t])
                dumh000 = (x + self.b_flag*(self.t_flag*(self.c1xb(actvs[1][t])+self.c2xb(actvs[2][t])+(self.fc1xb(actvs[3][t])).view(-1,1,100,100)))) * (1.+self.g_flag*self.t_flag*(self.c1xg(actvs[1][t])+self.c2xg(actvs[2][t])+(self.fc1xg(actvs[3][t])).view(-1,1,100,100)))
                actvs[0][t+1] = (F.relu(dumh000) - F.relu(dumh000-1))
                fb_acts[1][0][0][t] = self.l_flag*self.c1c1b(c1)
                fb_acts[1][0][1][t] = self.t_flag*self.c2c1b(actvs[2][t])
                fb_acts[1][0][2][t] = self.t_flag*(self.fc1c1b(actvs[3][t])).view(-1,self.n_feats,96,96)
                fb_acts_comb[1][0][t] = self.b_flag*(fb_acts[1][0][0][t] + fb_acts[1][0][1][t] + fb_acts[1][0][2][t])
                fb_acts[1][1][0][t] = self.l_flag*self.c1c1g(c1)
                fb_acts[1][1][1][t] = self.t_flag*self.c2c1g(actvs[2][t])
                fb_acts[1][1][2][t] = self.t_flag*(self.fc1c1g(actvs[3][t])).view(-1,self.n_feats,96,96)
                fb_acts_comb[1][1][t] = self.g_flag*(fb_acts[1][1][0][t] + fb_acts[1][1][1][t] + fb_acts[1][1][2][t])
                c1 = F.relu(self.conv1(actvs[0][t+1])+self.b_flag*(self.l_flag*self.c1c1b(c1)+self.t_flag*(self.c2c1b(actvs[2][t])+(self.fc1c1b(actvs[3][t])).view(-1,self.n_feats,96,96)))) * (1.+self.g_flag*(self.l_flag*self.c1c1g(c1)+self.t_flag*(self.c2c1g(actvs[2][t])+(self.fc1c1g(actvs[3][t])).view(-1,self.n_feats,96,96))))
                actvs[1][t+1] = self.pool(c1)
                fb_acts[2][0][0][t] = self.l_flag*self.c2c2b(c2)
                fb_acts[2][0][1][t] = self.t_flag*(self.fc1c2b(actvs[3][t])).view(-1,self.n_feats*2,28,28)
                fb_acts_comb[2][0][t] = self.b_flag*(fb_acts[2][0][0][t] + fb_acts[2][0][1][t])
                fb_acts[2][1][0][t] = self.l_flag*self.c2c2g(c2)
                fb_acts[2][1][1][t] = self.t_flag*(self.fc1c2g(actvs[3][t])).view(-1,self.n_feats*2,28,28)
                fb_acts_comb[2][1][t] = self.g_flag*(fb_acts[2][1][0][t] + fb_acts[2][1][1][t])
                c2 = F.relu(self.conv2(actvs[1][t+1])+self.b_flag*(self.l_flag*self.c2c2b(c2)+self.t_flag*(self.fc1c2b(actvs[3][t])).view(-1,self.n_feats*2,28,28))) * (1.+self.g_flag*(self.l_flag*self.c2c2g(c2)+self.t_flag*(self.fc1c2g(actvs[3][t])).view(-1,self.n_feats*2,28,28)))
                actvs[2][t+1] = self.pool(c2)
                fb_acts[3][0][0][t] = self.l_flag*self.fc1fc1b(actvs[3][t])
                fb_acts[3][1][0][t] = self.l_flag*self.fc1fc1g(actvs[3][t])
                fb_acts_comb[3][0][t] = self.b_flag*fb_acts[3][0][0][t]
                fb_acts_comb[3][1][t] = self.g_flag*fb_acts[3][1][0][t]
                actvs[3][t+1] = F.relu(self.fc1(actvs[2][t+1].view(-1, self.n_feats*2 * 9 * 9))+self.b_flag*self.l_flag*self.fc1fc1b(actvs[3][t])) * (1.+self.g_flag*self.l_flag*self.fc1fc1g(actvs[3][t]))
                actvs[4] = torch.cat((actvs[4],actvs[3][t+1]),1)
        actvs[5] = torch.log(torch.clamp(F.softmax(self.fc2(actvs[4]),dim=1),1e-10,1.0))
        return fb_acts_comb

# Defining the RNN class to be able to take perturbed feedback (incoming at T) and apply it to that sweep while keeping previous sweeps constant and naturally executing the subsequent sweeps
class RNNet_1step(nn.Module):
    def __init__(self, n_feats=8, ker_size=5,t_steps=3,b_flag=0,g_flag=1,l_flag=1,t_flag=1):
        super(RNNet_1step, self).__init__()
        self.conv1 = nn.Conv2d(1, n_feats, ker_size)
        self.pool = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(n_feats, n_feats*2, ker_size)
        self.fc1 = nn.Linear(n_feats*2 * 9 * 9, n_feats*16)
        self.fc2 = nn.Linear(n_feats*16*t_steps, 20)
        self.dropout = nn.Dropout(0.5)
        self.c1xb = nn.ConvTranspose2d(n_feats,1,7,3) # in_channel, out_channel, kernel_size, stride, padding
        self.c2xb = nn.ConvTranspose2d(n_feats*2,1,20,10)
        self.fc1xb = nn.Linear(n_feats*16, 100*100)
        self.c1c1b = nn.Conv2d(n_feats, n_feats, ker_size, 1, 2)
        self.c2c1b = nn.ConvTranspose2d(n_feats*2,n_feats,16,10)
        self.fc1c1b = nn.Linear(n_feats*16, 96*96*n_feats)
        self.c2c2b = nn.Conv2d(n_feats*2, n_feats*2, ker_size, 1, 2)
        self.fc1c2b = nn.Linear(n_feats*16, 28*28*n_feats*2)
        self.fc1fc1b = nn.Linear(n_feats*16, n_feats*16)
        self.c1xg = nn.ConvTranspose2d(n_feats,1,7,3) # in_channel, out_channel, kernel_size, stride, padding
        self.c2xg = nn.ConvTranspose2d(n_feats*2,1,20,10)
        self.fc1xg = nn.Linear(n_feats*16, 100*100)
        self.c1c1g = nn.Conv2d(n_feats, n_feats, ker_size, 1, 2)
        self.c2c1g = nn.ConvTranspose2d(n_feats*2,n_feats,16,10)
        self.fc1c1g = nn.Linear(n_feats*16, 96*96*n_feats)
        self.c2c2g = nn.Conv2d(n_feats*2, n_feats*2, ker_size, 1, 2)
        self.fc1c2g = nn.Linear(n_feats*16, 28*28*n_feats*2)
        self.fc1fc1g = nn.Linear(n_feats*16, n_feats*16)
        self.n_feats = n_feats
        self.t_steps = t_steps
        self.b_flag = b_flag
        self.g_flag = g_flag
        self.l_flag = l_flag
        self.t_flag = t_flag
    def forward(self, x, fb_acts_comb_org, fb_acts_comb_pert, pert_layer, pert_type, pert_time):
        fb_b = {}
        fb_g = {}
        for lay in np.arange(4):
            if pert_layer == lay:
                if pert_type == 0: # feed original feedback at selected layer
                    fb_b[lay] = fb_acts_comb_org[lay][0][pert_time]
                    fb_g[lay] = fb_acts_comb_org[lay][1][pert_time]
                elif pert_type == 1: # feed perturbed feedback at selected layer
                    fb_b[lay] = fb_acts_comb_pert[lay][0][pert_time]
                    fb_g[lay] = fb_acts_comb_pert[lay][1][pert_time]
                elif pert_type == 2: # feed control feedback at selected layer
                    fb_random_b = fb_acts_comb_pert[lay][0][pert_time].detach().clone()
                    fb_random_g = fb_acts_comb_pert[lay][1][pert_time].detach().clone()
                    for imgh in np.arange(list(fb_random_b.size())[0]):
                        fb_random_b_diff = fb_acts_comb_pert[lay][0][pert_time][imgh] - fb_acts_comb_org[lay][0][pert_time][imgh]
                        idx = torch.randperm(fb_random_b_diff.nelement())
                        fb_random_b[imgh] = fb_random_b_diff.view(-1)[idx].view(fb_random_b_diff.size()) + fb_acts_comb_org[lay][0][pert_time][imgh]
                        fb_random_diff = fb_acts_comb_pert[lay][1][pert_time][imgh] - fb_acts_comb_org[lay][1][pert_time][imgh]
                        idx = torch.randperm(fb_random_diff.nelement())
                        fb_random_g[imgh] = fb_random_diff.view(-1)[idx].view(fb_random_diff.size()) + fb_acts_comb_org[lay][1][pert_time][imgh]
                    fb_b[lay] = fb_random_b
                    fb_g[lay] = fb_random_g
            else: # feed original feedback at all other layers
                fb_b[lay] = fb_acts_comb_org[lay][0][pert_time]
                fb_g[lay] = fb_acts_comb_org[lay][1][pert_time]
        actvs = {}
        actvs[0] = {}
        actvs[1] = {}
        actvs[2] = {}
        actvs[3] = {}
        actvs[0][0] = F.relu(x) - F.relu(x-1)
        c1 = F.relu(self.conv1(actvs[0][0]))
        actvs[1][0] = self.pool(c1)
        c2 = F.relu(self.conv2(actvs[1][0]))
        actvs[2][0] = self.pool(c2)
        actvs[3][0] = F.relu(self.fc1(actvs[2][0].view(-1, self.n_feats*2 * 9 * 9)))
        actvs[4] = actvs[3][0]
        if self.t_steps > 0:
            for t in np.arange(self.t_steps-1):
                if t == pert_time:
                    dumh000 = (x + fb_b[0]) * (1.+fb_g[0])
                else:
                    dumh000 = (x + self.b_flag*(self.t_flag*(self.c1xb(actvs[1][t])+self.c2xb(actvs[2][t])+(self.fc1xb(actvs[3][t])).view(-1,1,100,100)))) * (1.+self.g_flag*self.t_flag*(self.c1xg(actvs[1][t])+self.c2xg(actvs[2][t])+(self.fc1xg(actvs[3][t])).view(-1,1,100,100)))
                actvs[0][t+1] = (F.relu(dumh000) - F.relu(dumh000-1))
                if t == pert_time:
                    c1 = F.relu(self.conv1(actvs[0][t+1])+fb_b[1]) * (1.+fb_g[1])
                else:
                    c1 = F.relu(self.conv1(actvs[0][t+1])+self.b_flag*(self.l_flag*self.c1c1b(c1)+self.t_flag*(self.c2c1b(actvs[2][t])+(self.fc1c1b(actvs[3][t])).view(-1,self.n_feats,96,96)))) * (1.+self.g_flag*(self.l_flag*self.c1c1g(c1)+self.t_flag*(self.c2c1g(actvs[2][t])+(self.fc1c1g(actvs[3][t])).view(-1,self.n_feats,96,96))))
                actvs[1][t+1] = self.pool(c1)
                if t == pert_time:
                    c2 = F.relu(self.conv2(actvs[1][t+1])+ fb_b[2]) * (1.+ fb_g[2])
                else:
                    c2 = F.relu(self.conv2(actvs[1][t+1])+self.b_flag*(self.l_flag*self.c2c2b(c2)+self.t_flag*(self.fc1c2b(actvs[3][t])).view(-1,self.n_feats*2,28,28))) * (1.+self.g_flag*(self.l_flag*self.c2c2g(c2)+self.t_flag*(self.fc1c2g(actvs[3][t])).view(-1,self.n_feats*2,28,28)))
                actvs[2][t+1] = self.pool(c2)
                if t == pert_time:
                    actvs[3][t+1] = F.relu(self.fc1(actvs[2][t+1].view(-1, self.n_feats*2 * 9 * 9))+ fb_b[3]) * (1.+ fb_g[3])
                else:
                    actvs[3][t+1] = F.relu(self.fc1(actvs[2][t+1].view(-1, self.n_feats*2 * 9 * 9))+self.b_flag*self.l_flag*self.fc1fc1b(actvs[3][t])) * (1.+self.g_flag*self.l_flag*self.fc1fc1g(actvs[3][t]))
                actvs[4] = torch.cat((actvs[4],actvs[3][t+1]),1)
        actvs[5] = torch.log(torch.clamp(F.softmax(self.fc2(actvs[4]),dim=1),1e-10,1.0))
        return actvs[5]

if __name__ == '__main__':

############################# NETWORK PARAMETERS ##################################

    n_feats = 8 # in Conv layer 1
    ker_size = 5 # in Conv layer 1
    b_h = 0 # bias/additive modulation flag
    g_h = 1 # gain/multiplicative modulation flag
    l_h = 1 # lateral interactions flag
    t_h = 1 # top-down interactions flag

    net_num = 5

    t_steps = 4 # number of timesteps

    net_save_str = 'rnn_bglt_'+str(b_h)+str(g_h)+str(l_h)+str(t_h)+'_t_'+str(t_steps)+'_num_'+str(net_num)
    print(net_save_str)

    n_ex = 1000
    n_rep = 5

    net_all = RNNet_all_fbr(n_feats,ker_size,4,b_h,g_h,l_h,t_h)
    net_all = net_all.float()
    net_all.load_state_dict(torch.load(net_save_str+'.pth',map_location=torch.device('cpu')))
    net_all.eval()

    net_fin = RNNet_1step(n_feats,ker_size,4,b_h,g_h,l_h,t_h)
    net_fin = net_fin.float()
    net_fin.load_state_dict(torch.load(net_save_str+'.pth',map_location=torch.device('cpu')))
    net_fin.eval()

############################# Perturbation analysis ##################################

    perturbed_accuracies = np.zeros([6,2,4,t_steps-1,n_rep]) # perturbation (pos_x/pos_y/orientation/scale/cat-within/cat-between), pert/control, layers, timesteps
    original_accuracy = np.zeros([n_rep,1])

    for repr in np.arange(n_rep):

        imgs_h,imgs_h_xswap,imgs_h_yswap,imgs_h_rotswap,imgs_h_sizeswap,imgs_h_catswap_w,imgs_h_catswap_b,labs_h,pos_x_h,pos_y_h,size_h,rot_h = gen_images(n_ex,2)

        input_img_org = torch.from_numpy(imgs_h).float()
        input_img_xswap = torch.from_numpy(imgs_h_xswap).float()
        input_img_yswap = torch.from_numpy(imgs_h_yswap).float()
        input_img_rotswap = torch.from_numpy(imgs_h_rotswap).float()
        input_img_sizeswap = torch.from_numpy(imgs_h_sizeswap).float()
        input_img_catswap_w = torch.from_numpy(imgs_h_catswap_w).float()
        input_img_catswap_b = torch.from_numpy(imgs_h_catswap_b).float()
        labels_img = torch.from_numpy(labs_h).float()

        fb_org = net_all(input_img_org)
        outputs = net_fin(input_img_org,fb_org,fb_org,0,0,2)

        _, predicted = torch.max(outputs.data, 1)
        total = labels_img.size(0)
        correct = np.sum(predicted.cpu().numpy() == torch.max(labels_img, 1)[1].cpu().numpy())
        original_accuracy[repr] = correct/total

        for th in np.arange(t_steps-1):
            for lay in np.arange(4):
                for pert in np.arange(6):

                    if pert == 0:
                        input_img_pert = input_img_xswap
                    elif pert == 1:
                        input_img_pert = input_img_yswap
                    elif pert == 2:
                        input_img_pert = input_img_rotswap
                    elif pert == 3:
                        input_img_pert = input_img_sizeswap
                    elif pert == 4:
                        input_img_pert = input_img_catswap_w
                    elif pert == 5:
                        input_img_pert = input_img_catswap_b

                    fb_pert = net_all(input_img_pert)

                    for contr in np.arange(2):

                        outputs = net_fin(input_img_org,fb_org,fb_pert,lay,contr+1,th)

                        _, predicted = torch.max(outputs.data, 1)
                        total = labels_img.size(0)
                        correct = np.sum(predicted.cpu().numpy() == torch.max(labels_img, 1)[1].cpu().numpy())
                        perturbed_accuracies[pert,contr,lay,th,repr] = perturbed_accuracies[pert,contr,lay,th,repr] + correct/total

                        print(repr,lay,pert,contr,th)

    out_str = 'fb_perturb-'+'rnn_bglt_'+str(b_h)+str(g_h)+str(l_h)+str(t_h)+'_t_'+str(t_steps)+'_num_'+str(net_num)+'.npy'
    with open(out_str, 'wb') as f:
        np.save(f, original_accuracy)
        np.save(f, perturbed_accuracies)
