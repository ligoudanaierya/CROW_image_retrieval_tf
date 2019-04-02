import numpy as np
import  os
import glob
import numpy as np
import tensorflow as tf
import vgg16
from scipy.misc import imread, imresize

from crow import apply_max_aggregation,normalize

images = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
sess = tf.Session()
vgg = vgg16.vgg16(images, 'vgg16_weights.npz', sess)
img = imread('data/oxford_003335.jpg',mode='RGB')
img =imresize(img,(224,224))
img = img.reshape((1, 224, 224, 3))
img1= imread('data/magdalen_000503.jpg',mode='RGB')
img1 = imresize(img1, (224, 224))
img1 = img1.reshape((1, 224, 224, 3))
img2= imread('data/oxford_001517.jpg',mode='RGB')
img2 = imresize(img2, (224, 224))
img2 = img2.reshape((1, 224, 224, 3))
feed_dict = {images: img}
feed_dict1 = {images: img1}
feed_dict2 = {images: img2}
pool = sess.run(vgg.pool5, feed_dict=feed_dict)
print(pool.shape)
pool1 = sess.run(vgg.pool5, feed_dict=feed_dict1)
pool2 = sess.run(vgg.pool5, feed_dict=feed_dict2)
feature1 = np.reshape(pool1, [7, 7, 512])
feature1 =feature1.transpose((2,0,1))
l1 = apply_max_aggregation(feature1)
l1 =normalize(l1)
feature2 = np.reshape(pool2, [7, 7, 512])
feature2 =feature2.transpose((2,0,1))
l2 = apply_max_aggregation(feature2)
l2 =normalize(l2)
feature = np.reshape(pool, [7, 7, 512])
feature =feature.transpose((2,0,1))
l = apply_max_aggregation(feature)
l =normalize(l)

print(((l-l1)**2).sum())
print(((l-l2)**2).sum())
