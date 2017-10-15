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
        imgs = filter(lambda x: x.endswith('jpg'), imgs_src)
        self.img_files = [img_dir + imgs for imgs in imgs_src]

        # self.msk_files = [f for f in glob.glob(msk_dir + '/*.bmp')]
        #self.img_files = [img_dir +'/'+ f.split('/')[-1].split('.')[0]+'.jpg' for f in self.msk_files]

    def __len__(self):
        return len(self.msk_files)

    def __getitem__(self, idx):
        im = cv2.imread(self.img_files[idx])
        im = cv2.resize(im, (0,0), fx=0.5, fy=0.5)
        im = cv2.resize(im, (512,512))
        im = np.transpose(im, (2, 0, 1)).astype(np.float32) / 255.
        mask = cv2.imread(self.msk_files[idx], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.
        mask = cv2.resize(mask, (0,0), fx=0.5, fy=0.5)
        mask = cv2.resize(mask, (512,512))
        mask = np.expand_dims(mask, axis=0)
        return im, mask


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
    criterion = hellinger_distance
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    print("preparing training data ...")
    train_set = LCDDataset("./imgs/", "./masks/")
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
            im = images.data[0].cpu().numpy()
            im = np.transpose(im, (1, 2, 0)) * 255
            im = im.astype(np.uint8)
            mask = outputs.data[0].cpu().numpy()[0, :, :] * 255
            mask = mask.astype(np.uint8)
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            save_im = np.concatenate((im, mask), axis=1)
            cv2.imwrite('./save_images/'+str(k)+'.jpg', save_im)
                
            vloss = criterion(outputs, masks, size_average=False)
            val_loss.update(vloss.data[0], images.size(0))

        print("Epoch {}, Loss: {}, Validation Loss: {}".format(epoch+1, train_loss.avg, val_loss.avg))

    if (epoch + 1) % 10 == 0:
        torch.save(net.state_dict(), './models/seg_{0}.p'.format(epoch+1))
    return net
'''
def test(model):
    model.eval()
'''




if __name__ == "__main__":
    print("begin tarin")
    net = train()
    torch.save(net.state_dict(), './models/seg_final.p')
    
