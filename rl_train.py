import numpy as np
from env import env
from PIL import Image
import os

def main():
    myenv = env()

    Data_dir = '../../Downloads/ffhq/facehq/'
    data_info = os.listdir(Data_dir)
    #data_info.sort()

    global_i = 0
    for path in data_info:
        path = os.path.join(Data_dir, path)
        #img = np.array(Image.open(path)) / 255.
        img = np.array(Image.open(path)) 
        myenv.reset(img)
        for n in range(100):
            done = myenv.step(n, global_i)
            if done is None:
                continue
            if done:
                break
        global_i += 1

    

if __name__ == "__main__":
    main()
