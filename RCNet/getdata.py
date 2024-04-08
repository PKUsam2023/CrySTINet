from torch.utils.data import Dataset
import torch


class MyDataset(Dataset):
    def __init__(self, mat, label):
        self.mat = mat
        self.label = label


    def __getitem__(self, idx):
        xrd = self.mat[idx]
        label = self.label[idx]

        xrd = torch.tensor(xrd).unsqueeze(dim=0).to(torch.float32)
        label = torch.tensor(label).to(torch.long)

        return xrd, label


    def __len__(self):
        return len(self.label)

