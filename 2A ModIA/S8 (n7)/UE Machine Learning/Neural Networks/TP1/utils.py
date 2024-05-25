
import scipy.io as sio
import numpy as np

def read_real_data(src):
    real_data = sio.loadmat(src+'.mat')
    pk = real_data['imgs']
    pk = np.transpose(pk, [2,0,1])
    print('read_real_data',pk.shape)
    return pk

def read_fake_data(src):
    fake_data = sio.loadmat(src+'.mat')
    pk = fake_data['imgs']
    print('read_fake_data,pk.shape)
    return pk
