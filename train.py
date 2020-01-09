from sys import argv
from PIL import Image
from model import *
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tfrc import get_input_data
import os
import get_data

from tensorflow.python.framework import graph_util
import json

batch_size = 2000
dropout = 0.3
learning_rate = 0.001
epochs = 1200000
size = 32
seq_len = size * size * 3
# read_log = argv[1] == 'log'
read_log = True


"""
def get_one_img_data(img_path):
    data = Image.open(img_path)
    data = data.convert('RGB')
    data = (np.array(data.getdata()) / 255).reshape((size, size, 3))
    #data = data[np.newaxis,:,:,:]
    return data


def get_data_info(img_dir):
    img_names = os.listdir(img_dir)

    x_paths = []
    for img_name in img_names:
            x_paths.append(img_dir + '/' + img_name)

    return x_paths, len(x_paths)


def data_generator(x_paths, i, l):
    x_batch = []
    y_batch = []
    j = 0
    while j < batch_size:
        if i >= l:
                i = 0
        x_batch.append(get_one_img_data(x_paths[i]))
        y_batch.append(int(x_paths[i].split('_')[0].split('/')[1]))
        j += 1
        i += 1

    return np.array(x_batch), np.array(y_batch), i
"""
Data_dir = '../../Downloads/ffhq/facehq/'
data_info = os.listdir(Data_dir)
data_info.sort()
train_list = data_info[:55000]
val_list = data_info[55000:len(data_info)]


model = build_network(learning_rate, size)
#prob = tf.nn.softmax(logits)

def filter_by_prob(x, y, sess, thres=0.2):
    px0 = x[y==0]
    my_prob = sess.run(model.prob, feed_dict={model.x: px0, model.dropout: 0, model.is_train: False})[:,1]
    similar_idx = np.argsort(my_prob)
    pos_x = px0[similar_idx[-260:-100]]

    #pos_x = px0[my_prob>thres]
    neg_x = x[y==1]
    print(len(pos_x), len(neg_x), my_prob[similar_idx[-10:]])

    #print(np.where(c1*c2 == True))
    #pos_x = x[c1*c2]
    #pos_x = x[]
    #c1 = y==1
    #c2 = my_prob<0.9
    #neg_x = x[c1*c2]

    new_x = pos_x.tolist() + neg_x.tolist()
    new_y = [0]*len(pos_x) + [1]*len(neg_x)

    return np.array(new_x), np.array(new_y)


if __name__ == "__main__":
    output_dir = "gen_output"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    saver=tf.train.Saver()

    init_global = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        #tf_writer = tf.summary.FileWriter('graph_log', sess.graph)
        sess.run(init_global)
        sess.run(init_local)
        if read_log:
            with open("log/checkpoint",'r') as f1:
                txt = f1.readline()
                point = txt.strip().replace('model_checkpoint_path: ','').replace("\"",'')
                saver.restore(sess,"log/%s"%point)

        step = 0
        i = 0
        test_i = 0

        #x_paths,  l = get_data_info('train')
        #test_x_paths, test_l = get_data_info('val')


        while step < epochs:
            step += 1
            i += 1
            test_i += 1
            if i >= len(train_list):
                i = 0
            if test_i >= len(val_list):
                test_i = 0
            x, label = get_data.get_batch_size(Data_dir + train_list[i])
#x, label = filter_by_prob(x, label, sess)
            #print('training data left:', len(x))

            test_x, test_label = get_data.get_batch_size(Data_dir + val_list[test_i])
            #print(test_i, ': ', train_list[i], val_list[test_i])
            #x = sess.run(trX)
            #x,label,i = data_generator(x_paths, i, l)
            #test_x, test_label, test_i = data_generator(test_x_paths, test_i, test_l)

            #x = np.reshape(x, [-1, size, size, 3])

            ##############	  define your input here	###########
            # x = np.random.random((100, size, size, 3))
            # label = np.random.randint(0,2,100)
            # test_x = np.random.random((100, size, size, 3))
            # test_label = np.random.randint(0,2,100)

            #if random.randint(0,1) == 1:
            #	x = np.array(list(map(np.fliplr, x.tolist())))

            sess.run(model.opt, feed_dict={model.x: x,  model.label: label, model.dropout: dropout, model.is_train: True})

            if step % 20 == 0:
                random.shuffle(train_list)
                random.shuffle(val_list)
                loss, acc = sess.run([model.loss, model.acc], feed_dict={model.x: x, model.label: label, model.dropout: dropout, model.is_train: False})
                val_loss, val_acc = sess.run([model.loss, model.acc], feed_dict={model.x: test_x, model.label: test_label, model.dropout: 0, model.is_train: False})
                #print("Epoch %d: %d/%d - loss: %.4f\tacc: %.4f" % (step * batch_size // l, step, l, loss, acc))
                print("%d/%d - loss: %.4f | %.4f\tacc: %.4f | %.4f" % (step, epochs, loss, val_loss, acc, val_acc))
                #print("Epoch %d: %d/%d - loss: %.4f | %.4f\tacc: %.4f | %.4f" % (step * batch_size, step, epochs, loss, val_loss, acc, val_acc))

                if step % 20 == 0:
                    checkpoint_filepath='log/step-%d.ckpt' % step
                    saver.save(sess,checkpoint_filepath)
                    print('checkpoint saved!')


                    #输出相似度较高的图片
                    # fake_x = np.random.random((100, size, size, 3))
                    fake_x = test_x[test_label==0]
                    print(test_i, ': ', val_list[test_i])
                    #print(fake_x.shape)
                    #print(test_x.shape)
                    fake_img_prob = sess.run(model.prob, feed_dict={model.x: fake_x, model.dropout: 0, model.is_train: False})

                    similar_idx = np.argsort(fake_img_prob[:,1])[-5:]
                    print(similar_idx)

                    g_sample = fake_x[similar_idx]
                    for ii, val in enumerate(g_sample):
                        img_arr = val * 255
                        im = Image.fromarray(img_arr.astype('uint8'))

                        prob = fake_img_prob[similar_idx[ii],1]
                        im.save("%s/%d_p%d_%d_pred.jpg" % (output_dir, step, 100*prob, ii))
