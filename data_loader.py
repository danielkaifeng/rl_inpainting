from PIL import Image
from sys import argv
import numpy as np
from tfrc import *

def get_img_data(image_path):
	im = Image.open(image_path) 
	im = im.convert('RGB')
	data = np.array(im.getdata())
	#data.shape (16384, 3)
	return data.flatten()


if __name__ == "__main__":
	filelist = argv[1]
	tf_filepath = "data/ocean.tfrecords"
	#writer = tf.io.TFRecordWriter("./data/orange_train.tfrecords")
	writer = tf.io.TFRecordWriter(tf_filepath)

	with open(filelist,'r') as f1:	
		txt = f1.readlines()
	for line in txt:
		image_path = line.strip()
		data = get_img_data(image_path)
		data = data/255.
		write_tfrecords(data, writer)
	#np.savetxt("data/train_data.txt", total_data, fmt='%.4f', delimiter=',')
	writer.close()

	print('~'*40)

	data = get_input_data(tf_filepath, 49152, 2)
	sess = tf.Session()
	for i in range(2):
		print(sess.run(data))
	sess.close()














