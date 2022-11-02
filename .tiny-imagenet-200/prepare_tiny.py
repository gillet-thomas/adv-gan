import csv
import os 
import cv2
import numpy as np

with open('.tiny-imagenet-200/words.txt', newline = '') as f:
    rows = ( line.rstrip('\n').split('\t') for line in f )
    imagenet_labels = { row[0]:row[1:] for row in rows }

lines = open('.tiny-imagenet-200/wnids.txt').read().splitlines()
tiny_labels = { line:imagenet_labels[line] for line in lines }

# print(len(tiny_labels))
# print(tiny_labels)

# n07920052 = espresso
# n07614500 = ice cream

images = []
folder = '.tiny-imagenet-200/train/' + 'n07920052' + '/images/'
for filename in os.listdir(folder):
    img = cv2.imread(os.path.join(folder,filename))
    if img is not None:
        images.append(img)

print(images[98].shape)
print(images[98].min())
print(images[98].max())
print(np.average(images[98]))
print(np.median(images[98]))

cv2.imshow('image', images[98])
cv2.waitKey(0)

img = cv2.normalize(images[80], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
resized = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)

print(img.min())
print(img.max())
print(np.average(img))
print(np.median(img))


cv2.imshow('image', images[98])
cv2.waitKey(0)



