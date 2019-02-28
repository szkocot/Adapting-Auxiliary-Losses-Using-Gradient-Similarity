import os, pickle
import h5py
import numpy as np # numerical operations
from tqdm import tqdm # pretty progress bar
import dask.array as da # compute global mean, std of rgb images

data_dir = '/home/szymon/gdrive/imagenet64'
files = [os.path.join(data_dir,x) for x in os.listdir(data_dir) if not ('.txt' in x or '.h5' in x)]
img_size = 64
img_size2 = img_size * img_size

dataset_size = 0
for fn in files:
    with open(fn, 'rb') as file:
        dataset_size += len(pickle.load(file)['data'])

with h5py.File(os.path.join(data_dir,'imagenet64_all.h5'), 'w') as f:
    dset_data = f.create_dataset("data", (dataset_size,img_size,img_size,3), dtype = 'uint8')
    dset_labels = f.create_dataset("labels", (dataset_size,), dtype = 'uint16')
    i = 0
    for fn in tqdm(files):
        with open(fn, 'rb') as file:
            d = pickle.load(file)
        data_temp = np.array(d['data'])
        data_temp = np.dstack(
            (data_temp[:, :img_size2], 
            data_temp[:, img_size2:2*img_size2], 
            data_temp[:, 2*img_size2:]))
        dset_data[i:i+len(data_temp)] = data_temp.reshape((data_temp.shape[0], img_size, img_size, 3))
        dset_labels[i:i+len(data_temp)] = np.array(d['labels'])
        i += len(data_temp)
    
with h5py.File(os.path.join(data_dir,'imagenet64_all.h5'), 'r') as f:    
    # compute global mean and std of images
    arr = da.from_array(f['data'], chunks=(1000, img_size, img_size, 3))
    print('Computing means')
    means = da.mean(arr, axis = (0,1,2)).compute()
    print('Computing stds')
    stds = da.std(arr, axis = (0,1,2)).compute()
    print('Finished!')

with h5py.File(os.path.join(data_dir,'imagenet64_all.h5'), 'a') as f:  
    dset_mean = f.create_dataset("mean", (3,))
    dset_std = f.create_dataset("std", (3,))
    dset_mean = means
    dset_std = stds