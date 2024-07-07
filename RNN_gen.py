# Script to define the RNN and dataset and to train the RNN
# Requires tensorflow 1.13, python 3.7, scikit-learn, and pytorch 1.6.0

############################# IMPORTING MODULES ##################################

import torch
from torchvision import datasets
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
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

# A function to generate images and the respective labels for training and testing
def gen_images(n_imgs,n_set): # n_imgs required, set used (0 train, 1 val, 2 test) 8 objects in image (1 is intact), 2 levels of zoom, rotation and x/y pos for each object
    imgs_h = np.zeros([n_imgs,1,100,100])
    imgs_h1 = np.zeros([n_imgs,1,100,100])
    labs_h = np.zeros([n_imgs,20])
    pos_x_h = np.zeros([n_imgs,2])
    pos_y_h = np.zeros([n_imgs,2])
    size_h = np.zeros([n_imgs,2])
    rot_h = np.zeros([n_imgs,2])
    n_objs = 8
    for n_im in np.arange(n_imgs):
        inst_img = np.zeros([100,100])
        inst_img1 = np.zeros([100,100])
        obj_ord = np.linspace(0,n_objs-1,n_objs)
        dum_obj_ind = 4+np.random.randint(n_objs/2)
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
            dumh111 = (np.random.random(1)[0] > 0.5)*1
            if dumh111 == 0: # zoom 0.9 or 1.5
                inst_obj = zoom(inst_obj,0.9+(np.random.random(1)[0]-0.5)/5.) # zoom 0.8 to 1.
            else:
                inst_obj = zoom(inst_obj,1.5+(np.random.random(1)[0]-0.5)/5.) # zoom 1.4 to 1.6
            if i == dum_obj_ind:
                size_h[n_im,dumh111] = 1.
            dumh111 = (np.random.random(1)[0] > 0.5)*1
            if dumh111 == 0: # rotate 30 or -30
                inst_obj = rotate(inst_obj,30+(np.random.random(1)[0]-0.5)*2*5,reshape=False) # rotate 25 to 35
            else:
                inst_obj = rotate(inst_obj,-30+(np.random.random(1)[0]-0.5)*2*5,reshape=False) # rotate -25 to -35
            if i == dum_obj_ind:
                rot_h[n_im,dumh111] = 1.
            if i != dum_obj_ind:
                inst_obj = im_scram(inst_obj,3) # scrambled if not object of interest
            if np.mod(obj_ord[i],4) == 0: # x_loc up or down
                x_loc = np.int(np.round(25 + (np.random.random(1)[0]-0.5)*2*2.5)) # 25 +- 2.5
                y_loc = np.int(np.round(25 + (np.random.random(1)[0]-0.5)*2*2.5)) # 25 +- 2.5
                if i == dum_obj_ind:
                    pos_y_h[n_im,0] = 1.
                    pos_x_h[n_im,0] = 1.
            elif np.mod(obj_ord[i],4) == 1:
                x_loc = np.int(np.round(75 + (np.random.random(1)[0]-0.5)*2*2.5)) # 75 +- 2.5
                y_loc = np.int(np.round(25 + (np.random.random(1)[0]-0.5)*2*2.5)) # 25 +- 2.5
                if i == dum_obj_ind:
                    pos_y_h[n_im,1] = 1.
                    pos_x_h[n_im,0] = 1.
            elif np.mod(obj_ord[i],4) == 2:
                x_loc = np.int(np.round(25 + (np.random.random(1)[0]-0.5)*2*2.5)) # 25 +- 2.5
                y_loc = np.int(np.round(75 + (np.random.random(1)[0]-0.5)*2*2.5)) # 75 +- 2.5
                if i == dum_obj_ind:
                    pos_y_h[n_im,0] = 1.
                    pos_x_h[n_im,1] = 1.
            elif np.mod(obj_ord[i],4) == 3:
                x_loc = np.int(np.round(75 + (np.random.random(1)[0]-0.5)*2*2.5)) # 75 +- 2.5
                y_loc = np.int(np.round(75 + (np.random.random(1)[0]-0.5)*2*2.5)) # 75 +- 2.5
                if i == dum_obj_ind:
                    pos_y_h[n_im,1] = 1.
                    pos_x_h[n_im,1] = 1.
            inst_obj = (inst_obj-np.min(inst_obj))/(np.max(inst_obj)-np.min(inst_obj))
            # print(np.int(np.floor(np.shape(inst_obj)[0]/2)),np.int(np.ceil(np.shape(inst_obj)[0]/2)),np.shape(inst_obj)[0])
            inst_img[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
            if i == dum_obj_ind:
                inst_img1[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] = (1-inst_obj)*inst_img1[x_loc-np.int(np.floor(np.shape(inst_obj)[0]/2.)):x_loc+np.int(np.ceil(np.shape(inst_obj)[0]/2.)),y_loc-np.int(np.floor(np.shape(inst_obj)[1]/2.)):y_loc+np.int(np.ceil(np.shape(inst_obj)[1]/2.))] + (inst_obj)*inst_obj
        inst_img = (inst_img-np.min(inst_img))/(np.max(inst_img)-np.min(inst_img))
        inst_img1 = (inst_img1-np.min(inst_img1))/(np.max(inst_img1)-np.min(inst_img1))
        if np.isnan(np.min(inst_img)) or np.isnan(np.min(inst_img1)):
            print('NaN in input')
            exit(1)
        imgs_h[n_im,0,:,:] = inst_img
        imgs_h1[n_im,0,:,:] = inst_img1
        labs_h[n_im,inst_lab] = 1.
    return imgs_h,imgs_h1,labs_h,pos_x_h,pos_y_h,size_h,rot_h

# Defining the RNN class
class RNNet_all(nn.Module):
    def __init__(self, n_feats=8, ker_size=5,t_steps=3,b_flag=1,g_flag=1,l_flag=1,t_flag=1):
        super(RNNet_all, self).__init__()
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
        actvsx = {}
        actvsc1 = {}
        actvsc2 = {}
        actvsfc = {}
        actvs[0] = actvsx
        actvs[1] = actvsc1
        actvs[2] = actvsc2
        actvs[3] = actvsfc
        actvs[0][0] = F.relu(x) - F.relu(x-1) # input activation clipping implemented this way to make it differentiable
        c1 = F.relu(self.conv1(actvs[0][0]))
        actvs[1][0] = self.pool(c1)
        c2 = F.relu(self.conv2(actvs[1][0]))
        actvs[2][0] = self.pool(c2)
        actvs[3][0] = F.relu(self.fc1(actvs[2][0].view(-1, self.n_feats*2 * 9 * 9)))
        actvs[4] = actvs[3][0]
        if self.t_steps > 0:
            for t in np.arange(self.t_steps-1):
                dumh000 = (x + self.b_flag*self.t_flag*(self.c1xb(actvs[1][t])+self.c2xb(actvs[2][t])+(self.fc1xb(actvs[3][t])).view(-1,1,100,100))) * (1.+self.g_flag*self.t_flag*(self.c1xg(actvs[1][t])+self.c2xg(actvs[2][t])+(self.fc1xg(actvs[3][t])).view(-1,1,100,100)))
                actvs[0][t+1] = (F.relu(dumh000) - F.relu(dumh000-1))
                c1 = F.relu(self.conv1(actvs[0][t+1])+self.b_flag*(self.l_flag*self.c1c1b(c1)+self.t_flag*(self.c2c1b(actvs[2][t])+(self.fc1c1b(actvs[3][t])).view(-1,self.n_feats,96,96)))) * (1.+self.g_flag*(self.l_flag*self.c1c1g(c1)+self.t_flag*(self.c2c1g(actvs[2][t])+(self.fc1c1g(actvs[3][t])).view(-1,self.n_feats,96,96))))
                actvs[1][t+1] = self.pool(c1)
                c2 = F.relu(self.conv2(actvs[1][t+1])+self.b_flag*(self.l_flag*self.c2c2b(c2)+self.t_flag*((self.fc1c2b(actvs[3][t])).view(-1,self.n_feats*2,28,28)))) * (1.+self.g_flag*(self.l_flag*self.c2c2g(c2)+self.t_flag*((self.fc1c2g(actvs[3][t])).view(-1,self.n_feats*2,28,28))))
                actvs[2][t+1] = self.pool(c2)
                actvs[3][t+1] = F.relu(self.fc1(actvs[2][t+1].view(-1, self.n_feats*2 * 9 * 9))+self.b_flag*self.l_flag*self.fc1fc1b(actvs[3][t])) * (1.+self.g_flag*self.l_flag*self.fc1fc1g(actvs[3][t]))
                actvs[4] = torch.cat((actvs[4],actvs[3][t+1]),1)
        actvs[5] = torch.log(torch.clamp(F.softmax(self.fc2(actvs[4]),dim=1),1e-10,1.0))
        return actvs

preprocess = transforms.Compose(
    [transforms.ToTensor()])

# Xavier intialisation
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

if __name__ == '__main__':

    # torch.autograd.detect_anomaly()

############################# NETWORK PARAMETERS ##################################

    n_feats = 8 # in Conv layer 1
    ker_size = 5 # in Conv layer 1
    b_h = 0 # bias modulation flag
    g_h = 1 # gain modulation flag
    l_h = 1 # lateral interactions flag
    t_h = 1 # top-down interactions flag

    net_num = 1 # to train multiple networks - id of current network

    batch_size = 32
    n_iter = 300000
    lrh = 0.0001
    t_steps = 4 # number of timesteps

    net_save_str = 'rnn_bglt_'+str(b_h)+str(g_h)+str(l_h)+str(t_h)+'_t_'+str(t_steps)+'_num_'+str(net_num)
    print(net_save_str)

    ############################# TRAINING ##################################

    train_loss = np.zeros([np.int(n_iter/100),1])
    val_loss = np.zeros([np.int(n_iter/1000),1])

    net = RNNet_all(n_feats,ker_size,t_steps,b_h,g_h,l_h,t_h)
    net.apply(weights_init)
    net = net.float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    # Gradient clipping at every parameter by registering hooks - the hook is called everytime a gradient is computed (https://pytorch.org/docs/stable/autograd.html)
    # To avaoid exploding gradients
    # for p in net.parameters():
    #     p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

    print('Net created!')

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(net.parameters(), lr=lrh)

    inputs_v,_,labels_v,_,_,_,_ = gen_images(5000,1)
    inputs_v = torch.from_numpy(inputs_v).float()
    inputs_v = inputs_v.to(device)
    labels_v = torch.from_numpy(labels_v).float()
    labels_v = labels_v.to(device)

    running_loss = 0.0

    for i in range(n_iter):

        inputs,_,labels,_,_,_,_ = gen_images(batch_size,0)
        inputs = torch.from_numpy(inputs).float()
        inputs = inputs.to(device)
        labels = torch.from_numpy(labels).float()
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs1 = net(inputs.float())
        loss = criterion(outputs1[5], torch.max(labels, 1)[1])

        loss.backward()
        # torch.nn.utils.clip_grad_value_(net.parameters(), clip_value)
        optimizer.step()

        running_loss += loss.item()

        if i == 0:
            print('%5d - loss: %.3f' % (i, running_loss))

        if i % 100 == 99:
            print('%5d - loss: %.3f' % (i, running_loss/100))
            train_loss[int((i+1)/100)-1,0] = running_loss/100
            running_loss = 0.0

        if i % 1000 == 999:

            net.eval()

            with torch.no_grad():

                outputs1 = net(inputs_v.float())
                outputs = outputs1[5]
                _, predicted = torch.max(outputs.data, 1)
                total = labels_v.size(0)
                correct = np.sum(predicted.cpu().numpy() == torch.max(labels_v, 1)[1].cpu().numpy())
                loss = criterion(outputs, torch.max(labels_v, 1)[1])
                val_loss[int((i+1)/1000)-1,0] = loss.item()

            print('%5d - Validation accuracy: %d %%' % (i, 100 * correct / total))
            print('%5d - Validation loss: %.3f' % (i, loss.item()))

            net.train()

        if i % 1000 == 999:
            np.savez('loss_'+net_save_str+'.npz', train_loss=train_loss, val_loss=val_loss)

        if i % 10000 == 9999:
            torch.save(net.state_dict(), net_save_str+'.pth')

    print('Done training!')

    net.eval()

    torch.save(net.state_dict(), net_save_str+'.pth')
    np.savez('loss_'+net_save_str+'.npz', train_loss=train_loss, val_loss=val_loss)
