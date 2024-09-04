import torch
import scipy.io as sio
import numpy as np
from torch.utils.data import Dataset, random_split

class DynDataset(Dataset):
    def __init__(self, dataset_path, config):
        super(DynDataset, self).__init__()
        self.dataset_path = dataset_path
        mat_contents = sio.loadmat(self.dataset_path)
        self.t = torch.tensor(mat_contents['T'], dtype=config.pt_dtype)
        self.x = torch.tensor(mat_contents['X'], dtype=config.pt_dtype)
        self.u = torch.tensor(mat_contents['U'], dtype=config.pt_dtype)
        self.x_dot = torch.tensor(mat_contents['X_dot'], dtype=config.pt_dtype)

    def __len__(self):
        return len(self.t)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        t = self.t[idx,:]
        x = self.x[idx,:]
        u = self.u[idx,:]
        x_dot = self.x_dot[idx,:]

        return t, x, u, x_dot
    
def get_test_and_training_data(dataset, train_ratio, further_train_ratio, seed_train_test, seed_actual_train):
    # Step 1: Split the dataset into training and testing
    torch.manual_seed(seed_train_test)
    np.random.seed(seed_train_test)
    print("==> Test-Train split: test_ratio = {:.02f}".format(1-train_ratio))
    total_train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - total_train_size
    total_train_dataset, test_dataset = random_split(dataset, [total_train_size, test_size])

    # Step 2: Further split the training dataset to use only a portion of it
    torch.manual_seed(seed_actual_train)
    np.random.seed(seed_actual_train)
    print("==> Further split: further_train_ratio = {:.02f}".format(further_train_ratio))
    actual_train_size = int(further_train_ratio * len(total_train_dataset))
    unused_size = len(total_train_dataset) - actual_train_size
    actual_train_dataset, _ = random_split(total_train_dataset, [actual_train_size, unused_size])

    return actual_train_dataset, test_dataset
