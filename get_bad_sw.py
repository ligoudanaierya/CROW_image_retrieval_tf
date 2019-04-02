import os

import cv2

from skimage import io

for root, dir, path in os.walk('badimage'):
    for f in path:
        name = os.path.splitext(os.path.basename(f))[0]
        print(name)
        img = cv2.imread(os.path.join('sw',name)+'.jpg')
        io.imsave(os.path.join('badsw', name) + '.jpg', img)