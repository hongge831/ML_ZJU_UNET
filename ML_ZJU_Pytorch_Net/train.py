from torch.autograd import Variable
from unet import UNET
import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import glob
import cv2
import os

num_epochs = 30
batch_size = 1

torch.manual_seed(42)

class LCDDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, msk_dir):
        msks_src = os.listdir(msk_dir)
        msks = filter(lambda x: x.endswith('bmp'), msks_src)
        self.msk_files = [msk_dir + msks for msks in msks_src]

        imgs_src = os.listdir(img_dir)
        imgs = filter(lambda x: x.endswith('bmp'), imgs_src)
        self.img_files = [img_dir + imgs for imgs in imgs_src]
        # self.msk_files = [f for f in glob.glob(msk_dir + '/*.bmp')]
        # self.img_files = [img_dir +'/'+ f.split('/')[-1].split('.')[0]+'.bmp'
        #                   for f in self.msk_files]

    def __len__(self):
        return len(self.msk_files)

    def __getitem__(self, idx):
        im = cv2.imread(self.img_files[idx], cv2.IMREAD_GRAYSCALE)
        height = im.shape[0]
        width = im.shape[1]
        im = cv2.copyMakeBorder(im,0,1024-height,0,128-width,cv2.BORDER_CONSTANT,value=0)
        # im = cv2.resize(im, (128, 1024))
		#when the image's size lager than 128,then make it to 128
        im = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
        im = im.astype(np.float32) / 255.
        im = np.expand_dims(im, axis=0)
        mask = cv2.imread(self.msk_files[idx], cv2.IMREAD_GRAYSCALE)
        p = list(np.nonzero(mask))
        ############################################
        p[0] = p[0] // 2
        p[1] = p[1] // 2
        heightM = mask.shape[0]
        widthM = mask.shape[1]
        mask = cv2.copyMakeBorder(mask, 0, 1024 - heightM, 0, 128 - widthM, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.resize(mask, (0, 0), fx=0.5, fy=0.5)
        # print(mask.shape)
        # mask = cv2.resize(mask, (128, 1024))
        mask[p] = 255
        mask = np.expand_dims(mask, axis=0).astype(np.float32) / 255.
        return im, mask

def euclid_distance(y_pred, y_true, size_average=True):
    n = y_pred.size(0)
    dif = y_true - y_pred
    dif = torch.pow(dif, 2)
    alpha = y_true * 0.95
    beta = (1 - y_true) * 0.05
    weight = alpha + beta
    dif = torch.sqrt(torch.sum(weight * dif))
    if size_average:
        dif = dif / n
    return dif


def hellinger_distance(y_pred, y_true, size_average=True):
    n = y_pred.size(0)
    dif = torch.sqrt(y_true) - torch.sqrt(y_pred)
    dif = torch.pow(dif, 2)
    dif = torch.sqrt(torch.sum(dif)) / 1.4142135623730951
    if size_average:
        dif = dif / n
    return dif

class Average(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count

def train():
    cuda = torch.cuda.is_available()
    net = UNET()
    if cuda:
        net = net.cuda()
    #criterion = hellinger_distance
    criterion = euclid_distance
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=0.01)

    print("preparing training data ...")
    train_set = LCDDataset("./train_imgs/", "./train_masks/")
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    print("done ...")

    test_set = LCDDataset("./test_imgs/", "./test_masks/")
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=4)
    for epoch in range(num_epochs):
        train_loss = Average()
        net.train()
        for i, (images, masks) in enumerate(train_loader):
            images = Variable(images)
            masks = Variable(masks)
            if cuda:
                images = images.cuda()
                masks = masks.cuda()

            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, masks, size_average=False)
            loss.backward()
            optimizer.step()
            train_loss.update(loss.data[0], images.size(0))

        val_loss = Average()
        net.eval()
        for k, (images, masks) in enumerate(test_loader):
            images = Variable(images)
            masks = Variable(masks)
            if cuda:
                images = images.cuda()
                masks = masks.cuda()

            outputs = net(images)

            # save to file
            for j in range(batch_size):
                im = images.data[j].cpu().numpy() * 255
                im = im[0,:,:].astype(np.uint8)
                msk = masks.data[j].cpu().numpy() * 255
                msk = msk[0,:,:].astype(np.uint8)
                mask = outputs.data[j].cpu().numpy()[0, :, :] * 255
                mask = mask.astype(np.uint8)
                save_im = np.concatenate((im, mask, msk), axis=1)
                cv2.imwrite('./save_images/'+str(k*batch_size + j)+'.jpg', save_im)
                
            vloss = criterion(outputs, masks, size_average=False)
            val_loss.update(vloss.data[0], images.size(0))

        print("Epoch {}, Loss: {}, Validation Loss: {}".format(epoch+1, train_loss.avg, val_loss.avg))

        if (epoch + 1) % 5 == 0:
            torch.save(net.state_dict(), 'models/seg_small_{0}.p'.format(epoch+1))
    return net

def test(model):
    model.eval()



if __name__ == "__main__":
    net = train()
    torch.save(net.state_dict(), 'models/seg_small_final.p')
    
