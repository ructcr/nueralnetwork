import numpy as np
weights=np.load('trained_weights.npy',allow_pickle=True)
biases=np.load('trained_biases.npy',allow_pickle=True)
print(weights.shape,biases.shape)
print(weights[3].shape)
