import numpy as np
import datetime
import os
import traceback
import shutil

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# from utils.radam import RAdam

import tqdm
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from utils.metric import accuracy


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Trainer(object):
    """base class for trainers"""

    def __init__(self):
        pass


class HEROS_Trainer(Trainer):
  def __init__(self, model, train_loader, val_loader,configs, wb = False):

    self.train_loader = train_loader
    self.val_loader = val_loader
    # self.test_loader = test_loader



    # self.configs = configs


    self.configs = configs

    self.batch_size = configs["batch_size"]
    self.learning_rate = configs["lr"]
    self.num_workers = configs["num_workers"]
    self.momentum = configs["momentum"]
    self.weight_decay = configs["weight_decay"]
    self.device = torch.device("cuda:0")
    self.max_plateau_count = configs["max_plateau_count"]
    self.max_epoch_num = configs["max_epoch_num"]
    # self.batch_size = 32
    # self.learning_rate = 0.0001
    # self.num_workers = 2
    # self.momentum = 0.9
    # self.weight_decay = 0.000001
    # self.device = torch.device("cuda:0")
    # self.max_plateau_count = 10
    # self.max_epoch_num = 20
    self.wb = wb

    self.model = model.to(self.device)
    

    self.train_loss_list = []
    self.train_acc_list = []
    self.val_loss_list = []
    self.val_acc_list = []
    self.best_train_acc = 0.0
    self.best_val_acc = 0.0
    self.best_train_loss = 0.0
    self.best_val_loss = 0.0
    self.test_acc = 0.0
    self.test_acc_ttau = 0.0
    self.plateau_count = 0
    self.current_epoch_num = 0

    # Set information for training
    self.start_time = datetime.datetime.now()

    self.checkpoint_dir = "/content/drive/MyDrive/LOL Highlight Detection/LOL recognition"

    self.checkpoint_path = os.path.join(self.checkpoint_dir, "{}_{}".format
                                        (self.configs["model"], self.start_time.strftime("%Y%b%d_%H.%M"),))


    self.train_ds = DataLoader(self.train_loader,batch_size=self.batch_size,num_workers=self.num_workers,
                    pin_memory=True, shuffle=True)
    self.val_ds = DataLoader(self.val_loader, batch_size = self.batch_size, num_workers=self.num_workers,
                    pin_memory=True, shuffle=False)
    


    
    self.criterion = nn.CrossEntropyLoss().to(self.device)


    self.optimizer = torch.optim.RAdam(
      params=self.model.parameters(),
      lr=self.learning_rate,
      weight_decay=self.weight_decay,
#       amsgrad = True,
    )
#     self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, nesterov=True, weight_decay=self.weight_decay)


    self.scheduler = ReduceLROnPlateau(
      self.optimizer,
      patience=3,
      min_lr=0,
      verbose=True,
      factor = 0.1,
    )
       
  def init_wandb(self):
    #set up wandb for training
    if self.wb == True:
      try:
        print("------------SETTING UP WANDB--------------")
        import wandb
        self.wandb = wandb
        self.wandb.login()
        print("------Wandb Init-------")

        self.wandb.init(
            project = "Heros trainer",
            name = self.configs["model"],
            config = {
                "batch_size" : self.batch_size,
                "learning_rate" : self.learning_rate,
                "epoch" : self.current_epoch_num
            }
        )
        # self.wandb.watch(self.model, self.criterion, log="all", log_freq=10)
        print()
        print("-----------------------TRAINING MODEL-----------------------")
      except:
          print("--------Can not import wandb-------")


  def step_per_train(self):

    self.model.train()
    train_loss = 0.0
    train_acc = 0.0

    for i, (images, labels) in tqdm.tqdm(
        enumerate(self.train_ds), total = len(self.train_ds), leave = True, colour = "blue", desc = f"Epoch {self.current_epoch_num}",
        bar_format="{desc}: {percentage:3.0f}%|{bar:50}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
    ):
      images = images.cuda(non_blocking = True)
      labels = labels.cuda(non_blocking = True)
    #   print("test")
      # compute output, accuracy and get loss
      y_pred = self.model(images)
    
      loss = self.criterion(y_pred, labels)
      acc = accuracy(y_pred, labels)[0]
      train_loss += loss.item()
      train_acc += acc.item()

      # compute gradient and do SGD step
      self.optimizer.zero_grad()
      loss.backward()
      self.optimizer.step()

      # write wandb
      metric = {
          " Loss" : train_loss / (i+1),
          " Accuracy" :train_acc / (i+1),
          " epochs" : self.current_epoch_num,
          " Learning_rate" : get_lr(self.optimizer)
      }
      if self.wb == True and i <= len(self.train_ds):
            self.wandb.log(metric)

      
    i += 1
    self.train_loss_list.append(train_loss / i)
    self.train_acc_list.append(train_acc / i)

    print(" Loss: {:.4f}".format(self.train_loss_list[-1]), ", Accuracy: {:.2f}%".format(self.train_acc_list[-1]))

  def step_per_val(self):
    self.model.eval()
    val_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
      for i, (images, labels) in tqdm.tqdm(
          enumerate(self.val_ds), total = len(self.val_ds), leave = True, colour = "green", desc = "        ",
          bar_format="{desc} {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
      ):
        images = images.cuda(non_blocking = True)
        labels = labels.cuda(non_blocking = True)

        # compute output, accuracy and get loss
        y_pred = self.model(images)

        loss = self.criterion(y_pred, labels)
        acc = accuracy(y_pred, labels)[0]

        val_loss += loss.item()
        val_acc += acc.item()

      i += 1
      self.val_loss_list.append(val_loss / i)
      self.val_acc_list.append(val_acc / i)

      print(" Val_Loss: {:.4f}".format(self.val_loss_list[-1]),", Val_Accuracy: {:.2f}%".format(self.val_acc_list[-1]))

      # write wandb
      if self.wb == True:
        metric = {
            " Val_Loss" : self.val_loss_list[-1],
            " Val_Accuracy" :self.val_acc_list[-1],
            # "Learning_rate" : self.learning_rate
        }
        self.wandb.log(metric)

  def Train_model(self):
    self.init_wandb()
    #self.scheduler.step(100 - self.best_val_acc)
    try:
      while not self.stop_train():
        self.update_epoch_num()
        self.step_per_train()
        self.step_per_val()

        self.update_state_training()

    except KeyboardInterrupt:
      traceback.print_exc()
      pass

    consume_time = str(datetime.datetime.now() - self.start_time)
    print("----------------------SUMMARY-----------------------")
    print(" After {} epochs and {} plateau count, consume {}".format((self.current_epoch_num), (self.plateau_count),consume_time[:-7]))
    print(" Best Accuracy on Train: {:.3f} ".format(self.best_train_acc))
    print(" Best Accuracy on Val: {:.3f} ".format(self.best_val_acc))
  #set up for training (update epoch, stopping training, write logging)
  def update_epoch_num(self):
    self.current_epoch_num += 1

  def stop_train(self):
    return (
        self.plateau_count > self.max_plateau_count or
        self.current_epoch_num > self.max_epoch_num
    )
  
  def update_state_training(self):
    if self.val_acc_list[-1] > self.best_val_acc:
      
      self.save_weights()
      self.plateau_count = 0
      self.best_val_acc = self.val_acc_list[-1]
      self.best_val_loss = self.val_loss_list[-1]
      self.best_train_acc = self.train_acc_list[-1]
      self.best_train_loss = self.train_loss_list[-1]
    else:
      self.plateau_count += 1
    self.scheduler.step(100 - self.best_val_acc)

  def save_weights(self):
    state_dict = self.model.state_dict()

    state = {
        "net": state_dict,
        "best_val_loss": self.best_val_loss,
        "best_val_acc": self.best_val_acc,
        "best_train_loss": self.best_train_loss,
        "best_train_acc": self.best_train_acc,
        "train_loss_list": self.train_loss_list,
        "val_loss_list": self.val_loss_list,
        "train_acc_list": self.train_acc_list,
        "val_acc_list": self.val_acc_list,
  
    }

    torch.save(state, self.checkpoint_path)