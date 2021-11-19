
# method 1
import h5py    # version >= 2.9.0

f = h5py.File("rotation_method1.h5", 'r')

for key in f.keys():
	print(f[key].name)
	print(f[key].shape)
	print(f[key].value)

# method 2
import numpy as np

data = np.fromfile("rotation_method2.dat", dtype=np.float32).reshape(3,3)
print(data)
