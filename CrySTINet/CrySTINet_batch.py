import sys
sys.path.append("../RCNet/")

import torch
from torch.utils.data import DataLoader

from network import RCNet
from getdata import MyDataset

import numpy as np
import pandas as pd
from tqdm import tqdm


class CrySTINet:

    """
    Batch test XRD patterns with all models
    """

    def __init__(self, model_kind_dict, subset = "test", alpha = 0.7, downsample=False) -> None:
        
        self.model_kind_dict = model_kind_dict
        self.subset = subset
        self.alpha = alpha
        self.downsample = downsample

        self.avg_xrd = np.load('../data/simulated/avg_xrd_mat.npy')
    

    def classify(self, dataset_path, model_name):

        # load data and use the same shuffle method as training
        dataset = np.load(dataset_path)
        np.random.seed(42)
        np.random.shuffle(dataset)

        # choose subset
        if self.subset == "train":
            dataset = dataset[:int(0.7*dataset.shape[0])]
        elif self.subset == "valid": 
            dataset = dataset[int(0.7*dataset.shape[0]): int(0.8*dataset.shape[0])]
        elif self.subset == "test":
            dataset = dataset[int(0.8*dataset.shape[0]): ]
        
        if self.downsample == True:
            dataset = self.sample(dataset, num=1000)

        # get raw label in 100 kinds
        label = dataset[:, -1]
        label = self.get_y_100(label, self.model_kind_dict[model_name])

        # calculate cos similarity
        num = np.dot(dataset[:, :-1], self.avg_xrd.T)
        denom = np.linalg.norm(dataset[:, :-1], axis=1).reshape(-1, 1) * np.linalg.norm(self.avg_xrd, axis=1)
        similarity_res = num / denom
        similarity_res = 0.5 + 0.5 * similarity_res

        # Model prediction & Obtain Similarity & calculate realiablity
        model_res = {}
        for idx in self.model_kind_dict.keys():

            # batch test XRD patterns with all models
            pred_y, conf = self.eval_batch(dataset, idx, kinds=len(self.model_kind_dict[idx]))
            pred_y_100 = self.get_y_100(pred_y, self.model_kind_dict[idx])
            
            # find the corresponding similarity from the similarity matrix
            similarity = []
            for i in range(similarity_res.shape[0]):
                similarity.append(similarity_res[i][pred_y_100[i]])

            # calculate realiablity
            res = (1-self.alpha) * conf + self.alpha * np.array(similarity)

            # save results to dict
            model_res[idx] = [len(self.model_kind_dict[idx]), pred_y, conf, pred_y_100, np.array(similarity), res]

        # output
        res_mat = np.zeros([len(model_res.keys()), similarity_res.shape[0]])
        pred_y_100_mat = np.zeros([len(model_res.keys()), similarity_res.shape[0]])
        for i, idx in enumerate(model_res.keys()):
            res_mat[i] = model_res[idx][5]
            pred_y_100_mat[i] = model_res[idx][3]
        
        end_res = np.zeros(similarity_res.shape[0])
        index = np.argmax(res_mat, axis=0)
        for i, j in enumerate(index):
            end_res[i] = pred_y_100_mat[j][i]

        # calculate CrySTINet accuracy
        acc = (end_res == label).astype(int).mean()

        return acc


    def eval_batch(self, dataset, model, kinds):

        net, device = self.load_model(model, kinds)

        x_test = dataset[:, :-1]
        y_test = dataset[:, -1]
        test_set = MyDataset(x_test, y_test)
        test_loader = DataLoader(test_set,
                                batch_size=512,
                                shuffle=False,
                                num_workers=16,
                                pin_memory=True)

        return self.eval(net, test_loader, device)


    def load_model(self, model, kinds):

        # load trained model
        device_ids = [0, 1, 2, 3]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(f"../RCNet/checkpoints/{model}_ckpt_best.pth")

        net = RCNet(n_classes=kinds)
        net = net.to(device)
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net.load_state_dict(checkpoint['net'], strict=True)
        net.eval()

        return net, device


    def eval(self, net, test_loader, device):

        pred_y = np.array([])
        conf = np.array([])

        for xrd, label in test_loader:

            xrd = xrd.to(device)
            label = label.to(device)

            out, confidence = net(xrd)
            tmp_pred_y = out.detach().argmax(dim=1).cpu().numpy()
            tmp_conf = confidence.detach().cpu().numpy().squeeze()

            pred_y = np.concatenate((pred_y, tmp_pred_y))
            conf = np.concatenate((conf, tmp_conf))

        return pred_y, conf


    def get_y_100(self, x, lst):
        """
        find the label in 100 kinds
        """
        tmp = []
        for each in x:
            tmp.append(lst[int(each)])
        return np.array(tmp)


    def sample(self, dataset: np.array, num=1000):
        '''
        downsample from each class of dataset
        '''
        sample_dataset = None
        y_test = dataset[:, -1]

        for kind in np.unique(y_test):
            
            index = np.argwhere(y_test == kind)
            dataset_kind = dataset[index].squeeze()

            if dataset_kind.shape[0] >= num:
                random_idx = np.random.choice(dataset_kind.shape[0], num, replace=False)
                if sample_dataset is None:
                    sample_dataset = dataset_kind[random_idx]
                else:
                    sample_dataset = np.concatenate((sample_dataset, dataset_kind[random_idx]))
            else:
                if sample_dataset is None:
                    sample_dataset = dataset_kind
                else:
                    sample_dataset = np.concatenate((sample_dataset, dataset_kind))

        return sample_dataset




if __name__ == '__main__':

    # ----------------
    # init param
    # ----------------
    # the trained models and structure types serial number they contain
    # the numbers are different from paper. 22~67 here are numbered 1~91 in sequence in the paper
    model_kind_dict = {
        # model_1
        'trained_model_1': [22, 32, 55, 62, 71, 74, 82, 83, 97],
        # model_2
        'trained_model_2': [ 2,  6, 18, 59, 87],
        # model_3
        'trained_model_3': [ 1, 15, 19, 33, 36, 37, 47, 52, 53, 76, 77, 90, 96],
        # model_4
        'trained_model_4': [ 4, 10, 12, 30, 34, 42, 56, 60, 69, 80, 81],   
        # model_5
        'trained_model_5': [21, 40, 41, 44, 45, 51, 64, 88, 89, 92, 94, 95, 98], 
        # model_6
        'trained_model_6': [7, 11, 16, 29, 38, 49, 50, 63, 68, 72],
        # model_7
        'trained_model_7': [0,  3,  8, 13, 14, 25, 73, 85],
        # model_8
        'trained_model_8': [20, 27, 28, 35, 46, 54, 84],
        # model_9  
        'trained_model_9': [ 5, 26, 39, 43, 70, 78, 79, 86, 91],
        # model_10
        'trained_model_10': [ 9, 17, 24, 31, 61, 67],  
    }

    subset = "test"     # selected subdataset (e.g. "train", "valid", "test")
    alpha = 0.7         # the proportion of similarity

    # -----------------
    # init CrySTINet
    # -----------------
    cstrnet = CrySTINet(model_kind_dict, subset, alpha)

    # -----------------
    # iterate over all 10 datasets and test accuracy
    # -----------------
    for num in tqdm(range(1, 11)):

        # load dataset
        dataset_path = f"../data/simulated/dataset/dataset_{num}.npy"
        model_name = f'trained_model_{num}'

        # calculate accuracy
        acc = cstrnet.classify(dataset_path, model_name)
        print(f"The accuracy on dataset_{num} is {acc}.")
