from sys import argv
from PIL import Image
from model import *
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tfrc import get_input_data
import os
import get_data
import matplotlib.pyplot as plt
from tensorflow.python.framework import graph_util
import json

batch_size = 8000
dropout = 0.3
learning_rate = 0.001
epochs = 100000
size = 32
seq_len = size * size * 3
read_log = argv[1] == 'log'


def test_image32(sess, img_path):
    src = Image.open(img_path)
    src = np.array(src)
    img = src.copy()
    for w in range(248, 600, 2):
        if w + 32 > 600:
            break
        cut = img[55:87, w:w+32]
        center = cut[8:24, 8:24]
        testx = []

        masks = []
        for i in range(batch_size):
            mh, mw = get_data.get_mask_position(img, 71, w + 16)
            mask = img[mh-8:mh+8, mw-8:mw+8]
            center = center * 0.5 + mask * 0.5
            masks.append(mask)
            fake = cut.copy()
            fake[8:24, 8:24] = center
            testx.append(fake)
        logits = sess.run(model.logits, feed_dict={model.x: np.array(testx), model.dropout: 0, model.is_train: False})
        results = logits[:,1] - logits[:,0]
        similar_idx = np.argsort(results)[-1:]
        print(similar_idx)
#final = testx[similar_idx[0]]
        masks = np.array(masks)
        final = masks[similar_idx]

        rand_h = random.randint(50, 76)
        rand_w = random.randint(w-13, w+13)
        src[rand_h+8:rand_h+24, rand_w+8:rand_w+24] = 0.8 * src[rand_h+8:rand_h+24, rand_w+8:rand_w+24]
        # src[71:103, w+8:w+24] = 0.9 * src[71:103, w+8:w+24]
        for fi in final:
            src[rand_h+8:rand_h+24, rand_w+8:rand_w+24] = 0.2 * fi + src[rand_h+8:rand_h+24, rand_w+8:rand_w+24]
            # src[71:103, w+8:w+24] = 0.1 * fi + src[71:103, w+8:w+24]
            #plt.imshow(src[0:64, w:w+64])
            #plt.show()

    src = Image.fromarray(src.astype('uint8')).convert('RGB')
    src.save('output/' + img_path.split('.')[0] + '_out_32.jpg')

"""
def test_image(sess, img_path):
    src = Image.open(img_path)
    src = np.array(src)
    img = src.copy()
    for w in range(248, 600, 2):
        if w + 64 > 600:
            break
        cut = img[55:119, w:w+64]
        center = cut[16:48, 16:48]
        testx = []

        masks = []
        for i in range(batch_size):
            mh, mw = get_data.get_mask_position(img, 87, w + 32)
            mask = img[mh-16:mh+16, mw-16:mw+16]
            center = center * 0.5 + mask * 0.5
            masks.append(mask)
            fake = cut.copy()
            fake[16:48, 16:48] = center
            testx.append(fake)
        logits = sess.run(model.logits, feed_dict={model.x: np.array(testx), model.dropout: 0, model.is_train: False})
        results = logits[:,1] - logits[:,0]
        similar_idx = np.argsort(results)[-10:-9]
        print(similar_idx)
#final = testx[similar_idx[0]]
        masks = np.array(masks)
        final = masks[similar_idx]

        rand_h = random.randint(68, 100)
        rand_w = random.randint(w-16, w+16)
        src[rand_h+16:rand_h+48, rand_w+16:rand_w+48] = 0.8 * src[rand_h+16:rand_h+48, rand_w+16:rand_w+48]
        # src[71:103, w+16:w+48] = 0.9 * src[71:103, w+16:w+48]
        for fi in final:
            src[rand_h+16:rand_h+48, rand_w+16:rand_w+48] = 0.2 * fi + src[rand_h+16:rand_h+48, rand_w+16:rand_w+48]
            # src[71:103, w+16:w+48] = 0.1 * fi + src[71:103, w+16:w+48]
            #plt.imshow(src[0:64, w:w+64])
            #plt.show()

    src = Image.fromarray(src.astype('uint8')).convert('RGB')
    src.save('output/' + img_path.split('.')[0] + '_out4.jpg')
"""

model = build_network(learning_rate, size)
#prob = tf.nn.softmax(logits)


if __name__ == "__main__":
    output_dir = "gen_output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    saver=tf.train.Saver()

    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    with tf.Session() as sess:
        #tf_writer = tf.summary.FileWriter('graph_log', sess.graph)
        sess.run(init_global)
        sess.run(init_local)
        if read_log:
            with open("log/checkpoint",'r') as f1:
                txt = f1.readline()
                point = txt.strip().replace('model_checkpoint_path: ','').replace("\"",'')
                saver.restore(sess,"log/%s"%point)

        test_image32(sess, '0000018.jpg')
