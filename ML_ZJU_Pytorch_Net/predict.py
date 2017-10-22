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
net.load_state_dict(torch.load('./80step_model/seg_small_final.p'))
net.cuda()
data = []
files = []
# for f in glob.glob(sys.argv[2] + '/*.bmp'):
#     print (f)
file_src = os.listdir('./test_imgs')
test_imgs = filter(lambda x: x.endswith('bmp'), file_src)
for f_test in test_imgs:
    f = './test_imgs/'+f_test
    files.append(f)
    im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im, (128, 1024))
    im2 = im.astype(np.float32) / 255.
    im2 = np.expand_dims(im2, axis=0)
    data.append(im2)
# print 'total images:', len(data)

# time1 = time.clock()
Data = np.stack(data)
num = len(data)

Data = Variable(torch.Tensor(Data), volatile=True)
Data = Data.cuda()
out = net(Data)

# time2 = time.clock()
# print 'time forward:', (time2 - time1) / num

Point = []
kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.uint8)
for k in range(num):
    msk = out.data[k].cpu().numpy() * 255
    msk = msk[0, :, :].astype(np.uint8)
    msk = cv2.GaussianBlur(msk, (3, 3), 1.)
    peak = cv2.dilate(msk, kernel)
    p = np.transpose(np.nonzero(msk > peak))
    p = [a for a in p if msk[a[0], a[1]] > 100]
    Point.append(p)

# time3 = time.clock()
# print 'time all:', (time3 - time1) / num

# draw results
for k in range(num):
    im_save = data[k][0, :, :] * 255
    im_save = im_save.astype(np.uint8)
    im_save = cv2.cvtColor(im_save, cv2.COLOR_GRAY2BGR)
    msk = out.data[k].cpu().numpy() * 255
    msk = msk[0, :, :].astype(np.uint8)
    msk = cv2.cvtColor(msk, cv2.COLOR_GRAY2BGR)
    for p in Point[k]:
        cv2.circle(im_save, (int(p[1]), int(p[0])), 8, (0, 255, 0), 1)
    title = files[k].split('/')[-1].split('.')[0]
    im3 = np.concatenate((im_save, msk), axis=1)
    cv2.imwrite('save_images/' + title + '.jpg', im3)

