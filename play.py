import numpy as np
import cv2
weights=np.load('trained_weights.npy',allow_pickle=True)
biases=np.load('trained_biases.npy',allow_pickle=True)
imgGrey=cv2.imread('digit.png',0)
imgGrey[imgGrey<30]=0
imgGrey[imgGrey>30]=255
imgGrey=cv2.resize(imgGrey,(28,28),interpolation=cv2.INTER_AREA)
cv2.imshow('imggrey',imgGrey)
cv2.waitKey()
imgGrey=imgGrey.reshape(1,784)
def feed_foward(data,weights,biases):
    d=data
    for w,b in zip(weights,biases):
        d=d.dot(w)+b
    print(d)
    return (np.argmax(d))
print(feed_foward(imgGrey,weights,biases))
