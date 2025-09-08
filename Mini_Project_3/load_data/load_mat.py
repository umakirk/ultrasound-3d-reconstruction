import scipy.io as sio
import numpy as np

mat_contents = sio.loadmat('givens/labels_LL1.mat')
array = mat_contents['labels'] #or other variable name

# reorder as necessary
print(array.shape)
reordered = np.transpose(array, (1,0,2))
print(reordered.shape)

np.save('labels_LL1', reordered)
