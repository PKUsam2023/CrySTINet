import sys
sys.path.append("../RCNet/")

import torch
from network import RCNet

import os
import numpy as np
import pandas as pd
from tqdm import tqdm


class CrySTINet:

    """
    Test a single xrd pattern from txt file with all models
    """

    def __init__(self, model_kind_dict, alpha = 0.7) -> None:
        
        self.model_kind_dict = model_kind_dict
        self.alpha = alpha

        self.avg_xrd = np.load('../data/simulated/avg_xrd_mat.npy')
    

    def classify(self, file_name):

        xrd = np.loadtxt(file_name)
        xrd = torch.tensor(xrd).unsqueeze(dim=0).unsqueeze(dim=0).to(torch.float32)

        # calculate cos similarity
        num = np.dot(xrd.squeeze(), self.avg_xrd.T)
        denom = np.linalg.norm(xrd) * np.linalg.norm(self.avg_xrd, axis=1)
        similarity_res = num / denom
        similarity_res = 0.5 + 0.5 * similarity_res

        # test a single xrd pattern with all models
        model_res = pd.DataFrame(index=self.model_kind_dict.keys(), columns=['kind', 'pred_y', 'conf'])
        for idx in model_res.index:
            pred_y, conf = self.eval(xrd, idx, kinds=len(self.model_kind_dict[idx]))
            model_res.loc[idx] = (len(self.model_kind_dict[idx]), pred_y, conf)

        # find the corresponding similarity from the similarity matrix
        similarity = []
        for idx in model_res.index:
            pred_y_100 = self.model_kind_dict[idx][model_res.at[idx, 'pred_y']]
            similarity.append(similarity_res[pred_y_100])
        model_res['similarity'] = similarity

        # calculate realiablity
        model_res['res'] = (1-self.alpha) * model_res['conf'] + self.alpha * model_res['similarity']

        return model_res


    def eval(self, xrd, model, kinds):

        # load trained model
        device_ids = [0, 1, 2, 3]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(f"../RCNet/checkpoints/{model}_ckpt_best.pth")

        net = RCNet(n_classes=kinds)
        net = net.to(device)
        net = torch.nn.DataParallel(net, device_ids=device_ids)
        net.load_state_dict(checkpoint['net'], strict=True)
        net.eval()

        out, conf = net(xrd)
        pred_y = out.detach().argmax(dim=1).item()

        return pred_y, conf.detach().cpu().item()




if __name__ == '__main__':

    # ----------------
    # init param
    # ----------------
    # the trained models and structure types they contain
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

    path = '../data/experiment/smooth_xrd/'         # XRD pattern after noise reduction
    file_lst = os.listdir(path)
    right_num = 0
    alpha = 0.7                                     # the proportion of similarity

    # -----------------
    # init CrySTINet
    # -----------------
    cstrnet = CrySTINet(model_kind_dict, alpha)

    # -----------------
    # iterate over all single XRD pattern and test accuracy
    # -----------------
    for file in tqdm(file_lst):

        file_name = path + file
        model_res_df = cstrnet.classify(file_name)

        # output
        last_model = model_res_df['res'].astype(float).idxmax()
        last_pred_y_100 = model_kind_dict[last_model][model_res_df.at[last_model, 'pred_y']]
        label = int(file.split('@')[2].split('_')[0][4:-1])
        print(f'The {file} was predicted as {last_pred_y_100} in all kinds.')

        if int(last_pred_y_100) == label:
            right_num += 1

    print(f"The accuracy is {(right_num / len(file_lst)):.4f}")
