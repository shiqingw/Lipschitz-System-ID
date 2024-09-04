import torch
import scipy.io as sio
from torch.utils.data import Dataset

class NODEDataset(Dataset):
    def __init__(self, dataset_path, config, subsmaple_ratio=1):
        super(NODEDataset, self).__init__()
        self.dataset_path = dataset_path
        mat_contents = sio.loadmat(self.dataset_path)
        T = torch.tensor(mat_contents['T'], dtype=config.pt_dtype)
        X = torch.tensor(mat_contents['X'], dtype=config.pt_dtype)
        for p in range(1, len(T)):
            if all(T[i] == T[i % p] for i in range(len(T))): 
                self.period = int(p)
                break
        self.t = T[0:self.period] # (period,)
        self.x0 = X[0:-1:self.period,:] # (N, n_state)
        self.x = torch.zeros((self.x0.shape[0], self.period, self.x0.shape[1]), dtype=config.pt_dtype) # (N, period, n_state)
        for i in range(self.x0.shape[0]):
            self.x[i,:,:] = X[i*self.period:(i+1)*self.period,:]
        if subsmaple_ratio > 1:
            self.t = self.t[0:-1:subsmaple_ratio]
            self.x = self.x[:,0:-1:subsmaple_ratio,:]

    def __len__(self):
        return len(self.x0)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        t = self.t
        x0 = self.x0[idx,:]
        x = self.x[idx,:,:]

        return t, x0, x