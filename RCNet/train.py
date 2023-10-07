import os
from time import time
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from getdata import MyDataset
from network import RCNet
from utils import *


class Model():

    def __init__(self, net, resume, checkpoint_model_path=None):

        # set save path
        self.root_path = './model_results/'
        self.dataset_name = 'dataset_1_partial'
        self.model_save_path = self.root_path
        self.figure_save_path = self.root_path
        self.dataset_path = f"../data/simulated/dataset/{self.dataset_name}.npy"
        self.board_writer_path = f'{self.root_path}tensorboard/'
        self.checkpoint_path = f'{self.root_path}checkpoints/'

        # create empty filefolds
        if not os.path.isdir(self.root_path):
            os.mkdir(self.root_path)
        if not os.path.isdir(self.model_save_path):
            os.mkdir(self.model_save_path)
        if not os.path.isdir(self.figure_save_path):
            os.mkdir(self.figure_save_path)
        if not os.path.isdir(self.board_writer_path):
            os.mkdir(self.board_writer_path)
        if not os.path.isdir(self.checkpoint_path):
            os.mkdir(self.checkpoint_path)

        # initialize tensorboard
        self.writer = SummaryWriter(self.board_writer_path)

        # checkpoint setting
        self.RESUME = resume                                     # resume training from checkpoint or not
        self.checkpoint_model_path = checkpoint_model_path       # checkpoint file path

        # create logger
        self.logger_path = f'{self.root_path}RCNet.log'
        self.logger = creat_log("RCNet", f"{self.logger_path}")
        self.logger.info("Initializing RCNet...")

        # initialize the training parameters
        self.num_works = 16
        self.batch_size = 1024
        self.trainset_rate = 0.7        # the proportion of train set to total data set
        self.validset_rate = 0.1        # the proportion of valid set to total data set
        self.testset_rate = 0.2         # the proportion of test set to total data set
        self.seed = 42
        # self.lr = 0.03
        self.lr = 0.001
        self.lr_last = 0.001
        self.start_epoch = -1
        self.EPOCH = 400
        self.warm_up = 5
        self.patience = 10
        self.budget = 0.6               # budget of confidence loss
        self.lmbda = 0.1                # initial coefficient of confidence loss

        self.logger.info(f'''
                        dataset: {self.dataset_name},
                        num_works: {self.num_works},
                        batch_size: {self.batch_size},
                        trainset_rate: {self.trainset_rate},
                        validset_rate: {self.validset_rate},
                        testset_rate: {self.testset_rate},
                        seed: {self.seed},
                        lr: {self.lr},
                        lr_last: {self.lr_last},
                        start_epoch: {self.start_epoch},
                        EPOCH: {self.EPOCH},
                        warm_up: {self.warm_up},
                        patience: {self.patience},
                        budget: {self.budget},
                        lmbda: {self.lmbda},
                        net: {net}
                        ''')
        
        # initialize the network
        self.device_ids = [0, 1, 2, 3]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net.to(self.device)
        self.net = torch.nn.DataParallel(self.net, device_ids=self.device_ids)
        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.lr)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        # recover model parameters from checkpoint
        if self.RESUME:

            try:
                checkpoint = torch.load(self.checkpoint_model_path)
            except:
                raise ValueError("Cannot find checkpoint file!")

            self.logger.info("Resume from checkpoint...")
            self.net.load_state_dict(checkpoint['net'], strict=True)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.start_epoch = checkpoint['epoch']


    def loss_fn(self, out, label, conf):

        exp = torch.exp(out)
        tmp1 = exp.gather(1, label.unsqueeze(-1)).squeeze()
        tmp2 = exp.sum(1)
        softmax = tmp1 / tmp2
        xentropy_loss = torch.mean(- torch.log(softmax))

        confidence_loss = torch.mean(- torch.log(conf.squeeze()))
        total_loss = xentropy_loss + self.lmbda * confidence_loss

        if self.budget > confidence_loss:
            self.lmbda = self.lmbda / 1.03
        elif self.budget <= confidence_loss:
            self.lmbda = self.lmbda / 0.8

        return total_loss


    def fit(self):

        # initialize
        min_loss = np.inf
        list_of_train_loss, list_of_train_acc = [], []
        list_of_valid_loss, list_of_valid_acc = [], []

        # load dataset
        self.logger.info("Loading dataset...")
        dataset = np.load(self.dataset_path)

        seed_everything(self.seed)
        np.random.shuffle(dataset)

        x_train = dataset[:int(self.trainset_rate*dataset.shape[0])][:, :-1]
        y_train = dataset[:int(self.trainset_rate*dataset.shape[0])][:, -1]
        x_valid = dataset[int(self.trainset_rate*dataset.shape[0]): int((self.trainset_rate+self.validset_rate)*dataset.shape[0])][:, :-1]
        y_valid = dataset[int(self.trainset_rate*dataset.shape[0]): int((self.trainset_rate+self.validset_rate)*dataset.shape[0])][:, -1]

        train_set = MyDataset(x_train, y_train)
        valid_set = MyDataset(x_valid, y_valid)

        train_loader = DataLoader(train_set,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=self.num_works,
                                pin_memory=True)
        valid_loader = DataLoader(valid_set,
                                batch_size=self.batch_size,
                                shuffle=True,
                                num_workers=self.num_works,
                                pin_memory=True)

        # train model
        self.logger.info("Training model...")
        for epoch in range(self.start_epoch+1, self.EPOCH):

            # decline in learning rate
            # for param_group in self.optimizer.param_groups:
                # param_group['lr'] = np.linspace(self.lr, self.lr_last, self.EPOCH)[epoch]

            t1 = time()

            train_loss, train_acc, train_conf = self.train(train_loader)
            valid_loss, valid_acc, valid_conf = self.valid(valid_loader)

            t2 = time()

            self.logger.info(f"epoch {epoch}: train_loss {train_loss:.4f}, valid_loss {valid_loss:.4f}; "
                            f"train_acc {train_acc:.4f}, valid_acc {valid_acc:.4f}; "
                            f"train_conf {train_conf:.4f}, valid_conf {valid_conf:.4f}; "
                            f"time {(t2 - t1):.1f} s")

            # save the output of each epoch
            list_of_train_loss.append(train_loss)
            list_of_valid_loss.append(valid_loss)
            list_of_train_acc.append(train_acc)
            list_of_valid_acc.append(valid_acc)

            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/Valid", valid_loss, epoch)
            self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
            self.writer.add_scalar("Accuracy/Valid", valid_acc, epoch)

            # save the best model
            if (valid_loss < min_loss) and (epoch > self.warm_up):
                min_loss = valid_loss
                checkpoint = {
                    "net": self.net.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": epoch
                }

                self.logger.info(f"best score: {min_loss:.4f} @ {epoch}")
                torch.save(checkpoint, f'{self.checkpoint_path}ckpt_best.pth')

        # save the final model
        torch.save(self.net.state_dict(), f"{self.model_save_path}model.pth")

        # plot
        self.logger.info("Plotting figure...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10 * 2, 8))

        ax1.plot(list_of_train_loss[1:], label='train loss')
        ax1.plot(list_of_valid_loss[1:], label='valid loss')
        ax1.legend()
        ax1.set_title("Loss Curve")

        ax2.plot(list_of_train_acc[1:], label='train acc')
        ax2.plot(list_of_valid_acc[1:], label='valid acc')
        ax2.legend()
        ax2.set_title("Acc Curve")

        fig.savefig(f"{self.figure_save_path}model_results.png", bbox_inches='tight')

        return 0


    def train(self, train_loader):

        self.net.train()

        epoch_loss = 0
        epoch_acc = 0
        epoch_conf = 0

        for xrd, label in tqdm(train_loader, desc='Training'):

            xrd = xrd.to(self.device)
            label = label.to(self.device)

            out, conf = self.net(xrd)
            label_one_hot = torch.zeros(out.shape).to(self.device).scatter_(1, label.unsqueeze(dim=1), 1)

            # Randomly set half of the confidences to 1 (i.e. no hints)
            b =  Variable(torch.bernoulli(torch.Tensor(conf.size()).uniform_(0, 1))).to(self.device)
            conff = conf * b + (1 - b)

            out_c = out * conff.expand_as(out) + label_one_hot * (1 - conff.expand_as(label_one_hot))
            out_c = torch.log(out_c)

            pred_y = out.detach().argmax(dim=1)
            acc = (pred_y == label).float().mean()

            self.optimizer.zero_grad()
            loss = self.loss_fn(out_c, label, conf)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_conf += conf.detach().cpu().numpy().mean()

        return epoch_loss / len(train_loader), epoch_acc / len(train_loader), epoch_conf / len(train_loader)


    def valid(self, valid_loader):

        self.net.eval()

        epoch_loss = 0
        epoch_acc = 0
        epoch_conf = 0

        for xrd, label in tqdm(valid_loader, desc='Validing'):

            xrd = xrd.to(self.device)
            label = label.to(self.device)

            out, conf = self.net(xrd)
            out_c = conf * out
            out_c = torch.log(out_c)

            pred_y = out.detach().argmax(dim=1)
            acc = (pred_y == label).float().mean()
            loss = self.loss_fn(out_c, label, conf)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            epoch_conf += conf.detach().cpu().numpy().mean()

        return epoch_loss / len(valid_loader), epoch_acc / len(valid_loader), epoch_conf / len(valid_loader)




if __name__ == '__main__':

    n_classes = 9       # for RCNet_1

    net = RCNet(n_classes)
    model = Model(net, resume=False)
    model.fit()
