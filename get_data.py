import cv2
import random
import numpy as np
import PIL.Image as Image


Size = 512
Csize = 32
Msize = 16
Cnum_per_img = 4000


def get_mask_position(img, h, w):
    mh = random.randint(Csize, Size-Csize-Msize)
    mw = random.randint(Csize, Size-Csize-Msize)
    if abs(h - mh + Msize//2) < 24 or abs(w - mw + Msize//2) < 24:
        mh, mw = get_mask_position(img, h, w)

    return mh, mw


def get_batch_size(img_path):
    batch_x = []
    batch_y = []
    img = Image.open(img_path)
    img = np.array(img)
    for _ in range(Cnum_per_img):
        h = np.random.randint(Csize, Size-Csize*2)
        w = np.random.randint(Csize, Size-Csize*2)
        cut = img[h:h+Csize, w:w+Csize]  # truth data

        batch_y.append(1)
        batch_x.append(np.array(cut) / 255.0)
        """
        if _ % 5 == 0:
            batch_y.append(1)
            batch_x.append(np.array(cut) / 255.0)
        """

        fake = cut.copy()  # 开始制作bad data
        center = fake[Msize//2:Msize+Msize//2, Msize//2:Msize+Msize//2]

        # 获取合理的mask坐标
        mh, mw = get_mask_position(img, h, w)
        mask = img[mh:mh+Msize, mw:mw+Msize]
        center = mask * 0.5 + center * 0.5
        fake[Msize//2:Msize+Msize//2, Msize//2:Msize+Msize//2] = center

        batch_x.append(np.array(fake) / 255.0)
        batch_y.append(0)

    return np.array(batch_x), np.array(batch_y)

