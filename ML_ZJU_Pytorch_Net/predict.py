import sys
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from unet import UNET
import glob
import time
import os

net = UNET()
net.load_state_dict(torch.load('./models/seg_final.p'))
net.cuda()
time1 = time.clock()
count = 0
file_src = os.listdir('./test_imgs')
test_imgs = filter(lambda x: x.endswith('jpg'), file_src)
for f_test in test_imgs:
    f = './test_imgs/'+f_test
    #print(f)
    im = cv2.imread(f)
    im = cv2.resize(im, (512, 512))
    im2 = np.transpose(im, (2, 0, 1)).astype(np.float32) / 255.
    im2 = np.expand_dims(im2, axis=0)
    im2 = Variable(torch.Tensor(im2))
    im2 = im2.cuda()
    out = net(im2)
    out = out.data[0].cpu().numpy() * 255
    out = out[0, :, :].astype(np.uint8)
    out = cv2.GaussianBlur(out, (5, 5), 1.5) > 127
    contours = cv2.findContours(out.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    im_copy = im.copy()
    cv2.drawContours(im, contours[1], -1, (0,0,255), 1)
    #I dont know the second parameter in func 'drawContours',it is wait for me to figure out
    #print(contours[1])

    im = np.concatenate((im_copy, im), axis=1)
    title = f.split('/')[-1].split('.')[0]
    cv2.imwrite('./save_images/' + title + '.jpg', im)
    count += 1

time2 = time.clock()
#print 'average time:', (time2-time1) / count



