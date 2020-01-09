import tensorflow as tf
from network import agent
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
import tensorflow as tf
import random
from random import randint
import cv2
from sys import argv

width = 128
height = 128
mem_len = 64
agent = agent()
plt.ion()
read_log = argv[1] == 'log'
sess = tf.Session()
init_global = tf.global_variables_initializer()
init_local = tf.local_variables_initializer()

sess.run(init_global)
sess.run(init_local)
saver=tf.train.Saver()

def switch(a,b):
    if a>b:
        tmp = a
        a = b
        b = tmp
    return a,b

class env():
    def __init__(self):
        img_size = 512
        self.epsilon = 0.5
        self.memory = []
#self.action = self.rpn(self.img, self.x)
#self.whole_dist = self.cal_dis()

        os.makedirs("rl_log", exists_ok=True)
        if read_log:
            with open("rl_log/checkpoint",'r') as f1:
                txt = f1.readline()
                point = txt.strip().replace('model_checkpoint_path: ','').replace("\"",'')
                saver.restore(sess,"rl_log/%s"%point)
                print('check loaded from :', point)

    def reset(self, img):
        self.gamma = 0.2
        self.img = img
        width = 128
        height = 128
        #xst = np.random.randint(20, 300-width)
        #yst = np.random.randint(50, 400-width)
        #self.img_mask = img[xst:xst+height, yst:yst+width]
        area = [[0,136],[370,136],[180,50],[180,362]]
        ar = area[np.random.randint(0,4)]

        self.img_mask = img[ar[0]:ar[0]+width, ar[1]:ar[1]+width] 
        self.gt_mask = self.img_mask.copy() 

        self.pos = self.make_dirty()


        self.dist = None

    def make_dirty(self):
        img = self.img_mask
        size = img.shape
        max_width = 8
        number = random.randint(2, 4)

        pos = []
        for _ in range(number):
            model = random.random()
            rc = [255, random.randint(150,230), random.randint(0,200)]
            if 1 or model < 0.6:
                # Draw random lines
                x1, x2 = randint(1, size[1]), randint(1, size[1])
                while x1 == x2:
                    x1, x2 = randint(1, size[1]), randint(1, size[1])
                y1, y2 = randint(1, size[0]), randint(1, size[0])
                while y1 == y2:
                    y1, y2 = randint(1, size[0]), randint(1, size[0])
                thickness = randint(4, max_width)
                cv2.line(img, (x1, y1), (x2, y2), rc, thickness)
                if x1 > x2:
                    x1, x2 = switch(x1, x2)                    
                if y1 > y2:
                    y1, y2 = switch(y1, y2)
                pos.append([x1, x2, y1, y2])

            elif 0 and model > 0.6 and model < 0.8:
                # Draw random circles
                x1, y1 = randint(1, size[1]), randint(1, size[2])
                radius = randint(4, max_width)
                cv2.circle(img, (x1, y1), radius, rc, -1)

            elif model > 0.6:
                # Draw random ellipses
                x1, y1 = randint(1, size[1]), randint(1, size[0])
                s1, s2 = randint(1, size[1]), randint(1, size[0])
                a1, a2, a3 = randint(3, 180), randint(3, 180), randint(3, 180)
                thickness = randint(4, max_width)
                cv2.ellipse(img, (x1, y1), (s1, s2), a1, a2, a3, rc, thickness)
                pos.append([np.max(x1-s1, 0), x1+s2, np.max(y1-s2, 0), y1+s2])

        return pos



    def repair(self, action):
        img_mask = self.img_mask.copy()

        idx = np.random.choice(list(range(len(self.pos))))
        pos = self.pos[idx]
        self.x = np.random.randint(pos[0],pos[1])
        self.y = np.random.randint(pos[2],pos[3])

        x_mask, y_mask, weight, height, alpha = action
        #x = np.int32(x * img_mask.shape[1])
        #y = np.int32(y * img_mask.shape[0])
        x2 = np.int32(x_mask * img_mask.shape[1])
        y2 = np.int32(y_mask * img_mask.shape[0])

        alpha = 0.5 + 0.5*alpha
        weight = np.int32(weight * img_mask.shape[1] * 0.2)
        height = np.int32(height * img_mask.shape[0] * 0.2)

        xe = self.x + weight
        ye = self.y + height
        x2e = x2 + weight
        y2e = y2 + height

        if xe > img_mask.shape[1] or x2e > img_mask.shape[1] or ye > img_mask.shape[0] or y2e > img_mask.shape[0]:
            #print('out of boundary')
            return None
        else:
            img_mask[self.y:ye, self.x:xe] = alpha * img_mask[y2:y2e, x2:x2e] + (1-alpha) * img_mask[self.y:ye, self.x:xe]
    
        if self.n % 10 == 0:
            #print("x_mask, y_mask, weight, height, alpha\n",  x2, y2, weight, height, alpha)
            print(x2, y2, weight, height, alpha)

        return img_mask

    def step(self, n, gi=0):
        self.gamma = gamma = 1/(1 + gi/200) 
        self.n = n
        img_mask = self.img_mask.copy()
        #mask = np.array(img_mask)  
        mask = np.array(img_mask) / 255. 
        if n < 10:
            action = np.random.random(5)
        elif 1 or n< 400:
                action = sess.run(agent.action, feed_dict={agent.mask: [mask]})[0]
                action += gamma * np.random.random(5)
                action /= (1+gamma)
        else:
            action = sess.run(agent.action, feed_dict={agent.mask: [mask]})[0]


        img_mask = self.repair(action)
        if img_mask is None:
            return None
    
        done, reward = self.observation(img_mask)
        #if reward > 0 or n == 0:
        if 1:
                plt.axis('off')
                #plt.imshow(np.uint8(img_mask*255))
                plt.imshow(img_mask)
                plt.show()
                plt.pause(0.01)
        #elif random.randint(0,5) == 1:
        #    self.memory.append([img_mask, reward, action])

        if reward > 0:
            self.memory.append([mask, reward, action])
            self.img_mask = img_mask.copy()
            #mask = np.array(img_mask) / 255. 

        if done:
            print('done!')

        _, q_val = sess.run([agent.opt, agent.q_val], feed_dict={agent.mask: [mask], agent.reward: [reward]})
        #q_val = sess.run(agent.q_val, feed_dict={agent.mask: [mask], agent.reward: [reward]})
        #if n % 100 == 0 or reward > 0:
        #    _, q_val = sess.run([agent.p_opt, agent.q_val], feed_dict={agent.mask: [img_mask]})
        #    sess.run(agent.q_opt, feed_dict={agent.mask: [img_mask], agent.reward: [reward]})
        if len(self.memory) >= mem_len:
            for n in range(10):
                idx = np.random.choice(list(range(mem_len)), 32)
                memory = np.array(self.memory)
                rpm = memory[idx]
                b_mask = [x[0] for x in rpm]
                b_reward = [x[1] for x in rpm]
                b_action = [x[2] for x in rpm]

                #b_mask = np.random.random((32, width, height, 3))
                #b_action = np.random.random((32,5))
                #b_mask = np.array(b_mask) / 255.
                #b_mask *= 255.
                ra_loss, _ = sess.run([agent.ra_loss, agent.ra_opt], feed_dict={agent.mask: b_mask, 
                                                                                                agent.reward_action: b_action, 
                                                                                                agent.reward: b_reward
                                                                                                })
                if n % 2 == 0:
                    print('ra_loss:', ra_loss)
                    action = sess.run(agent.action, feed_dict={agent.mask: b_mask})
            print('------------------> make action get more reward')
            action = sess.run(agent.action, feed_dict={agent.mask: b_mask})[:5]
            print(np.round(action, 2))
            print('------------------> make action get more reward')
            self.memory = self.memory[32:]

            #print(b_mask[0][:10][:10])
            #print(type(b_mask[0]))


        #self.gamma *= 0.95
        if reward > 0:
            print('n_step: %d, qval: %.4f, reward: %.4f, gamma: %.3f, delta: %.4f' % (n, q_val[0], reward, self.gamma, np.abs(q_val[0]-reward)))

        if n % 100 == 0:
            print("len of memory:", len(self.memory))
            print('n_step: %d, qval: %.4f, reward: %.4f, delta: %.4f' % (n, q_val[0], reward, np.abs(q_val[0]-reward)))
        if n % 100 == 0:
            if random.randint(0,10) == 0:
                checkpoint_filepath='rl_log/step-%d.ckpt' % n 
                saver.save(sess,checkpoint_filepath)
                print('checkpoint saved!')

        return done


    def observation(self, img_mask):
        dist = self.cal_dis(img_mask, self.gt_mask)
        #图像是否修复得比上次好
        reward = self.cal_reward(dist)

        #图像是否已经修复得足够好
        if dist < self.epsilon:
            done = True
            reward += 20
        else:
            done = False

        return done, reward

    # compare the similarity between patch and gt
    def cal_dis(self, img1, img2):
        return np.mean(np.square(img1 - img2))

    # 对于整个要修复的区域，是否比上一步修复得更好 
    def cal_reward(self, dist):
        if self.dist is None:
            self.dist = dist  
            reward = 0
#elif self.n < 200 or 
        elif dist < self.dist: # + self.gamma:
            reward = 10 * (self.dist - dist) # + self.gamma)
            #print('dist:', dist, self.dist, reward)
            assert reward > 0
            self.dist = dist
            #else:
            #    reward = self.dist - dist + self.gamma
        else:
            reward = 2 * (self.dist - dist) 
        #reward -= 0.002

        return reward


if __name__ == "__main__":
    myenv = env()

    ori_img = np.array(Image.open('ori_face512.jpg'))
    #img = np.array(Image.open('face512.jpg'))
    myenv.reset(ori_img)

    for n in range(20000):
        myenv.step(n)

    sess.close()
