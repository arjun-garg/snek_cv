import os
import copy
import torch
import random
import pickle
import argparse
import torchvision
import numpy as np
import torch.nn as nn

import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.utils.checkpoint as cp

from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score
from efficientnet_pytorch import EfficientNet
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler


model_loc = os.path.join("/", "home", "aniket17133", "snek","models")
MODEL_DIR = os.path.join("/", "home", "aniket17133", "snek","models")
DATASET_PATH = os.path.join("/", "home", "aniket17133", "snek","datasets", "train_small_10_cropped")

def load_dataset(root,
                 batchsize,
                 input_size,
                 crop_size,
                 validation_split=.2,
                 shuffle_dataset=True,
                 random_seed=42,
                 num_workers=2):

    data_path = root
    trainTransform  = torchvision.transforms.Compose([torchvision.transforms.Resize((input_size, input_size)),
                        torchvision.transforms.CenterCrop(crop_size),
                        torchvision.transforms.ToTensor(),
                        transforms.Normalize([0.0432, 0.0554, 0.0264], [0.8338, 0.8123, 0.7803]),
                        ])

    dataset = torchvision.datasets.ImageFolder(
        root=data_path,
        transform=trainTransform
    )

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))

    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batchsize,
                                               sampler=train_sampler,
                                               num_workers=num_workers)

    valid_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=num_workers)

    return train_loader, valid_loader


class EfficientNetClassifier(nn.Module):
    def __init__(self, n_classes, model_name="efficientnet-b5"):
        super(EfficientNetClassifier, self).__init__()

        self.effnet =  EfficientNet.from_pretrained(model_name)

        self.l1 = nn.Linear(1000 , 256)
        self.dropout = nn.Dropout(0.5)

        self.l2 = nn.Linear(256,n_classes) # 6 is number of classes
        self.relu = nn.LeakyReLU()

    def forward(self, input):
        x = self.effnet(input)
        x = x.view(x.size(0),-1)

        x = self.dropout(self.relu(self.l1(x)))
        x = self.l2(x)

        return x


num_epochs = 20
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Using device: {}".format(device))

model_name="efficientnet-b5"
# model = EfficientNet.from_pretrained(model_name)
model = EfficientNetClassifier(n_classes=10)
image_size = EfficientNet.get_image_size(model_name)

data = datasets.ImageFolder(DATASET_PATH)
num_classes = len(data.classes)

print("Number of classes: {}".format(num_classes))

train_loader, valid_loader = load_dataset(DATASET_PATH, batch_size, image_size, image_size)

dataloaders_dict = {
    "train": train_loader,
    "valid": valid_loader
}

model = model.to(device)
# Optimizer
# optimizer = optim.Adam(model.parameters(), lr=3e-4)
optimizer = optim.SGD(model.parameters(), lr=3e-4, momentum=0.9)

# Loss Funciton
criterion = nn.CrossEntropyLoss()

def train_model(model, dataloaders, criterion, optimizer, model_name, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_fscore = 0.0
    
    loss_train_evo=[]
    acc_train_evo=[]
    fs_train_evo=[]
    
    loss_val_evo=[]
    acc_val_evo=[]
    fs_val_evo=[]

    corrects_total = {}
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            fscore = []
            model_preds = []
            corrects = []

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # inputs, labels = inputs.cuda(), labels.cuda()
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                labels_cpu = labels.cpu().numpy()
                predictions_cpu = preds.cpu().numpy()
                Fscore = f1_score(labels_cpu, predictions_cpu, average='macro')
                fscore.append(Fscore)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                model_preds.extend(predictions_cpu)
                corrects.extend(labels_cpu)

            epoch_loss = running_loss / (len(dataloaders[phase]) * batch_size)
            epoch_acc = running_corrects.double() / (len(dataloaders[phase]) * batch_size)
            epoch_fscore = np.average(np.array(fscore))

            corrects_total["{}_{}".format(phase, epoch)] = {
                "model_preds": model_preds,
                "corrects": corrects
            }
            
            print('{} Loss: {:.4f} Acc: {:.4f} F: {:.3f}'.format(phase, epoch_loss, epoch_acc, epoch_fscore))

            MODEL_PATH = os.path.join(MODEL_DIR, model_name + "_model_{}.pt".format(epoch))
            OPTIM_PATH = os.path.join(MODEL_DIR, model_name + "_optim_{}.pt".format(epoch))

            torch.save(model.state_dict(), MODEL_PATH)
#             torch.save(optimizer.state_dict(), OPTIM_PATH)
            
            if phase == 'train':
                loss_train_evo.append(epoch_loss)
                epoch_acc = epoch_acc.cpu().numpy()
                acc_train_evo.append(epoch_acc)
                fs_train_evo.append(epoch_fscore)                
            else:
                loss_val_evo.append(epoch_loss)
                epoch_acc = epoch_acc.cpu().numpy()
                acc_val_evo.append(epoch_acc)
                fs_val_evo.append(epoch_fscore) 
                
            if phase == 'valid' and epoch_fscore > best_fscore:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model, loss_train_evo, acc_train_evo, fs_train_evo, loss_val_evo, acc_val_evo, fs_val_evo, corrects_total

model_ft, loss_train, acc_train, fs_train, loss_val, acc_val, fs_val, corrects_total = train_model(model,
                                                                                                   dataloaders_dict,
                                                                                                   criterion,
                                                                                                   optimizer,
                                                                                                   model_name,
                                                                                                   num_epochs=num_epochs)

def plot_metric(metric_train, metric_val, title):
    fig, (ax) = plt.subplots(1, 1)
    fig.suptitle(title)
    ax.set(xlabel='epoch')
    ax.plot(metric_train, label='Training')
    ax.plot(metric_val, label='Validation')
    ax.legend(loc='upper left')
    plt.savefig(os.path.join(MODEL_DIR, title + "_metric.png"))

plot_metric(loss_train, loss_val, 'Loss')
plot_metric(acc_train, acc_val, 'Accuracy')
plot_metric(fs_train, fs_val, 'F-Score')

torch.save(loss_train, os.path.join(MODEL_DIR, "loss_train.pkl"))
torch.save(loss_val, os.path.join(MODEL_DIR, "loss_val.pkl"))
torch.save(acc_train, os.path.join(MODEL_DIR, "acc_train.pkl"))
torch.save(acc_val, os.path.join(MODEL_DIR, "acc_val.pkl"))
torch.save(fs_train, os.path.join(MODEL_DIR, "fs_train.pkl"))
torch.save(fs_val, os.path.join(MODEL_DIR, "fs_val.pkl"))
torch.save(corrects_total, os.path.join(MODEL_DIR, "corrects_total.pkl"))

