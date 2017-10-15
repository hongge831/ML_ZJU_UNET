import os

imgs_src = os.listdir('./imgs')
imgs = filter(lambda x: x.endswith('jpg'), imgs_src)
list=['./imgs' + imgs for imgs in imgs_src]
list2 = ['./imgs' + '/' + f for f in list]
print(list)