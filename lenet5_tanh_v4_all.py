#!/usr/bin/env python
# coding: utf-8

# # Implementing SPSA for LeNet-5 in PyTorch

# ## Setup

import traceback
from itertools import cycle
from pathlib import Path

import seaborn
import os

import numpy as np
from datetime import datetime 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import datasets, transforms

from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt

from PSGD import CLROptimizer_pert, CLROptimizer

from sklearn.metrics import confusion_matrix, classification_report
import math

# check device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'




# parameters
RANDOM_SEED = 42
BATCH_SIZE = 32
test_batch_size = 1000
N_EPOCHS = 20
N_EPOCHS_P = 5
N_CLASSES = 10


# ## Helper Functions
# Function for splitting training data into training and validation sets
def get_train_valid_loader(random_seed, valid_size=0.1, shuffle=True):
    error_msg = '[!] valid_size should be in the range [0, 1].'
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])

    train_dataset = datasets.MNIST(
        root='../data', train=True, download=False, transform=transform)
    valid_dataset = datasets.MNIST(
        root='../data', train=True, download=False, transform=transform)

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=test_batch_size, sampler=valid_sampler)

    return train_dataset, train_loader, valid_loader, valid_dataset

# Function to loading the testing dataset
def get_test_loader(shuffle=True):
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])

    test_dataset = datasets.MNIST(
        root='../data', train=False, download=True, transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=shuffle)

    return test_loader, test_dataset

def get_accuracy(model, data_loader, device):
    '''
    Function for computing the accuracy of the predictions over the entire data_loader
    '''
    
    correct_pred = 0 
    n = 0
    
    with torch.no_grad():
        model.eval()
        for X, y_true in data_loader:

            X = X.to(device)
            y_true = y_true.to(device)

            _, y_prob = model(X)
            _, predicted_labels = torch.max(y_prob, 1)

            n += y_true.size(0)
            correct_pred += (predicted_labels == y_true).sum()

    return correct_pred.float() / n

def plot_results(plots_dir, runs, result_name,max_epoch):
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    imgpath = plots_dir / f'{result_name}.png'
    if os.path.exists(imgpath):
        os.remove(imgpath)
    
    if not imgpath.is_file():
        try:
            plt.figure(figsize=(16, 9))
            c_cycler = cycle(seaborn.color_palette())
            for name, results in runs.items():
                c = next(c_cycler)
                if result_name in results and len(results[result_name]):
                    train_batches = results['train_data_len'] // results['batch_size']

                    plt.plot(range(1,max_epoch+1),
                             results[result_name],
                             '.-', label=name, color=c)
            plt.xlabel('Epoch')
            plt.title(result_name)
            plt.legend()
            plt.savefig(imgpath)
        except Exception:
            traceback.print_exc()
    
def train(train_loader, model, criterion, optimizer, device):
    '''
    Function for the training step of the training loop
    '''

    model.train()
    running_loss = 0
    
    for X, y_true in train_loader:

        if isinstance(optimizer, CLROptimizer) or isinstance(optimizer, CLROptimizer_pert):
            model.zero_grad()          
        else:
            optimizer.zero_grad()
            
        X = X.to(device)
        y_true = y_true.to(device)
    
        # Forward pass
        y_hat, _ = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

        # Backward pass
        loss.backward()
        if isinstance(optimizer, CLROptimizer) or isinstance(optimizer, CLROptimizer_pert):
            optimizer.step(X, y_true, loss)          
        else:
            optimizer.step()
        
    epoch_loss = running_loss / len(train_loader.dataset)
    
    return model, optimizer, epoch_loss

def validate(valid_loader, model, criterion, device):
    '''
    Function for the validation step of the training loop
    '''
   
    model.eval()
    running_loss = 0
    
    for X, y_true in valid_loader:
    
        X = X.to(device)
        y_true = y_true.to(device)

        # Forward pass and record loss
        y_hat, _ = model(X) 
        loss = criterion(y_hat, y_true) 
        running_loss += loss.item() * X.size(0)

    epoch_loss = running_loss / len(valid_loader.dataset)
        
    return model, epoch_loss

def training_loop(model, criterion, optimizer, train_loader, valid_loader, epochs, device, epochs_pert, batch_size, print_every=1):
    '''
    Function defining the entire training loop
    '''
    
    # set objects for storing metrics
    train_losses = []
    valid_losses = []
    min_loss = np.Inf
    results = {'ep': [], 'train_losses': [], 'valid_losses': [], 'train_acc': [], 'valid_acc': [], 'ml': min_loss, 'train_data_len': len(train_loader.dataset), 'batch_size':batch_size }
    #best = {'ep': 0, 'train_loss': min_loss, 'valid_loss': min_loss, 'train_acc': 0, 'valid_acc': 0}
    if isinstance(optimizer, CLROptimizer):
        opt_save = 'spsa'            
    else:
        opt_save = 'sgd'
    # Train model
    for epoch in range(0, epochs):

        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer, device)
        results['train_losses'].append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            results['valid_losses'].append(valid_loss)

        if epoch % print_every == (print_every - 1):
            
            train_acc = get_accuracy(model, train_loader, device=device)
            results['train_acc'].append(train_acc)
            valid_acc = get_accuracy(model, valid_loader, device=device)
            results['valid_acc'].append(valid_acc)    
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')
        if train_loss <= results['ml']:
            #print('\tValidation loss decreased ({:.6f} --> {:.6f}).  Saving model ...\n'.format(results['ml'],valid_loss))
            print('\tTraining loss decreased ({:.6f} --> {:.6f}).  Saving model ...\n'.format(results['ml'],train_loss))
            results['ml'] = train_loss
            best1 = {'ep': epoch, 'train_loss': train_loss, 'valid_loss': valid_loss, 'train_acc': 100 * train_acc, 'valid_acc': 100 * valid_acc}
            torch.save(model.state_dict(), 'model_{}.pt'.format(opt_save))


    # Train model perturbation
    print('Epoch perturbation')

    if isinstance(optimizer, CLROptimizer):
        optimizer_pert = CLROptimizer_pert(model, criterion)
    else:
        optimizer_pert = optimizer
    results['ml'] = best1['valid_loss']
    for epoch in range(epochs, epochs+epochs_pert):
        model.load_state_dict(torch.load('model_{}.pt'.format(opt_save)))
        # training
        model, optimizer, train_loss = train(train_loader, model, criterion, optimizer_pert, device)
        results['train_losses'].append(train_loss)

        # validation
        with torch.no_grad():
            model, valid_loss = validate(valid_loader, model, criterion, device)
            results['valid_losses'].append(valid_loss)

        if epoch % print_every == (print_every - 1):
            
            train_acc = get_accuracy(model, train_loader, device=device)
            results['train_acc'].append(train_acc)
            valid_acc = get_accuracy(model, valid_loader, device=device)
            results['valid_acc'].append(valid_acc)    
                
            print(f'{datetime.now().time().replace(microsecond=0)} --- '
                  f'Epoch: {epoch}\t'
                  f'Train loss: {train_loss:.4f}\t'
                  f'Valid loss: {valid_loss:.4f}\t'
                  f'Train accuracy: {100 * train_acc:.2f}\t'
                  f'Valid accuracy: {100 * valid_acc:.2f}')

        if valid_loss <= results['ml']:
            print('\tValid loss decreased ({:.6f} --> {:.6f}).  Saving model ...\n'.format(results['ml'],valid_loss))            
            results['ml'] = valid_loss
            best2 = {'ep': epoch, 'train_loss': train_loss, 'valid_loss': valid_loss, 'train_acc': 100 * train_acc, 'valid_acc': 100 * valid_acc}
            torch.save(model.state_dict(), 'model_{}.pt'.format(opt_save))
            
    
    return results, best1, best2

def weight_reset(m):
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.clf()
    plt.ylabel('True label', fontdict = {'fontsize' : 20})
    plt.xlabel('Predicted label', fontdict = {'fontsize' : 18})
    plt.grid(False)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)    
    plt.title(title)
    plt.imshow(cm, cmap=plt.cm.jet, interpolation='nearest')

    for i, cas in enumerate(cm):
        for j, count in enumerate(cas):
            if count > 0:
                    xoff = .07 * len(str(count))
                    plt.text(j - xoff, i + .2, int(count), fontsize=9, color='white')

    #plt.show()
    plt.savefig('./plots/confusion_matrices.png')
def shapeGrid(n):
    width = math.ceil(math.sqrt(n))
    if width*(width-1)>=n:
        return [width,width-1]
    else:
        return [width,width]
    
class LeNet5(nn.Module):

    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )


    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs



torch.manual_seed(RANDOM_SEED)

model = LeNet5(N_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()

# Load training, validation, and testing data
train_dataset, train_loader, valid_loader, valid_dataset = get_train_valid_loader(RANDOM_SEED)
test_loader, test_dataset = get_test_loader()

runs = {}
###########################################################################################################

print('=====run SPSA ======')
model.apply(weight_reset)
optimizer = CLROptimizer(model, criterion)

runs['SPSA'], best1, best2 = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE, N_EPOCHS_P, BATCH_SIZE)

# Use model with the lowest validation set loss on test data
print('Epoch best: {}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Train accuracy: {:.2f}, Valid accuracy: {:.2f}'.format(best1['ep'],best1['train_loss'],best1['valid_loss'],best1['train_acc'],best1['valid_acc']))
print('Epoch best pertrbation: {}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Train accuracy: {:.2f}, Valid accuracy: {:.2f}'.format(best2['ep'],best2['train_loss'],best2['valid_loss'],best2['train_acc'],best2['valid_acc']))

model.load_state_dict(torch.load('model_spsa.pt'))
with torch.no_grad():
    mo, valid_loss = validate(valid_loader, model, criterion, DEVICE)

valid_acc = get_accuracy(model, valid_loader, device=DEVICE)

print('SPSA with best epoch:  Valid Loss: {:.4f},  Valid accuracy: {:.2f}'.format(valid_loss, 100*valid_acc))

with torch.no_grad():
    mo, test_loss = validate(test_loader, model, criterion, DEVICE)

test_acc = get_accuracy(model, test_loader, device=DEVICE)

print('SPSA with best epoch:  Test Loss: {:.4f},  Test accuracy: {:.2f}'.format(test_loss, 100*test_acc))

###########################################################################################################
print('=====run Adam ======')
model.apply(weight_reset)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

runs['Adam'], best1, best2 = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE, N_EPOCHS_P, BATCH_SIZE)

# Use model with the lowest validation set loss on test data
print('Epoch best: {}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Train accuracy: {:.2f}, Valid accuracy: {:.2f}'.format(best1['ep'],best1['train_loss'],best1['valid_loss'],best1['train_acc'],best1['valid_acc']))
print('Epoch best pertrbation: {}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Train accuracy: {:.2f}, Valid accuracy: {:.2f}'.format(best2['ep'],best2['train_loss'],best2['valid_loss'],best2['train_acc'],best2['valid_acc']))
model.load_state_dict(torch.load('model_sgd.pt'))
with torch.no_grad():
    mo, test_loss = validate(test_loader, model, criterion, DEVICE)

test_acc = get_accuracy(model, test_loader, device=DEVICE)

print('Adam with best epoch:  Test Loss: {:.4f},  Test accuracy: {:.2f}'.format(test_loss, 100*test_acc))

###########################################################################################################
print('=====run Adagrad ======')
model.apply(weight_reset)
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
runs['Adagrad'], best1, best2 = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE, N_EPOCHS_P, BATCH_SIZE)

# Use model with the lowest validation set loss on test data
print('Epoch best: {}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Train accuracy: {:.2f}, Valid accuracy: {:.2f}'.format(best1['ep'],best1['train_loss'],best1['valid_loss'],best1['train_acc'],best1['valid_acc']))
print('Epoch best pertrbation: {}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Train accuracy: {:.2f}, Valid accuracy: {:.2f}'.format(best2['ep'],best2['train_loss'],best2['valid_loss'],best2['train_acc'],best2['valid_acc']))
model.load_state_dict(torch.load('model_sgd.pt'))
with torch.no_grad():
    mo, test_loss = validate(test_loader, model, criterion, DEVICE)

test_acc = get_accuracy(model, test_loader, device=DEVICE)

print('Adagrad with best epoch:  Test Loss: {:.4f},  Test accuracy: {:.2f}'.format(test_loss, 100*test_acc))

###########################################################################################################
print('=====run SGD ======')
model.apply(weight_reset)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)#, momentum=0.5
runs['SGD'], best1, best2 = training_loop(model, criterion, optimizer, train_loader, valid_loader, N_EPOCHS, DEVICE, N_EPOCHS_P, BATCH_SIZE)

# Use model with the lowest validation set loss on test data
print('Epoch best: {}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Train accuracy: {:.2f}, Valid accuracy: {:.2f}'.format(best1['ep'],best1['train_loss'],best1['valid_loss'],best1['train_acc'],best1['valid_acc']))
print('Epoch best pertrbation: {}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Train accuracy: {:.2f}, Valid accuracy: {:.2f}'.format(best2['ep'],best2['train_loss'],best2['valid_loss'],best2['train_acc'],best2['valid_acc']))
model.load_state_dict(torch.load('model_sgd.pt'))
with torch.no_grad():
    mo, test_loss = validate(test_loader, model, criterion, DEVICE)

test_acc = get_accuracy(model, test_loader, device=DEVICE)

print('SGD with best epoch:  Test Loss: {:.4f},  Test accuracy: {:.2f}'.format(test_loss, 100*test_acc))

###############################################Comparing ############################################################



plots_dir_name = 'plots'
plot_results(plots_dir_name, runs, 'train_losses',N_EPOCHS+N_EPOCHS_P)
plot_results(plots_dir_name, runs, 'valid_losses',N_EPOCHS+N_EPOCHS_P)
plot_results(plots_dir_name, runs, 'train_acc',N_EPOCHS+N_EPOCHS_P)
plot_results(plots_dir_name, runs, 'valid_acc',N_EPOCHS+N_EPOCHS_P)

###############################################Matrix conf ############################################################

model.eval()

y_pred = []
y_true = []

# iterate over test data
for inputs, labels in test_loader:
        output, _ = model(inputs) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth

# constant for classes
classes = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')

# Build confusion matrix
cf_matrix = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10,6))#(10,10)
plot_confusion_matrix(cf_matrix, classes)

print(
    "Classification report for classifier :\n"
    f"{classification_report(y_true, y_pred, digits=3)}\n"
)

###############################################Prediction ############################################################
with torch.no_grad():
    indexes = np.random.randint(0,len(test_dataset),12)
    #print(indexes)

    # figure out the size of figure
    n = len(indexes)
    w,l = shapeGrid(n)

    #imgarrayX, imgarrayY = imgarray

    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    for i, index in zip(range(n),indexes):
        item = test_dataset[index]
        image = item[0].unsqueeze(0)#add .unsqueeze(0) for cnn
        true_target = item[1]
        # Generate prediction
        prediction, _ = model(image)
        
        # Predicted class value using argmax
        predicted_class = np.argmax(prediction)
        
        # Reshape image
        image = image.reshape(32, 32, 1)            
        plt.subplot(w, l, i+1)
        s = "True: {}, Pred: {}".format(true_target, predicted_class)
        plt.title(s, fontdict = {'fontsize' : 10})
        plt.axis('off')
        plt.imshow(image, cmap='gray')
    plt.savefig("./plots/predect.png")
    plt.show() 

