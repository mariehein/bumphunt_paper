import os
import MAF_sample_from_file_import as sample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import yaml
import numpy as np
from sklearn.utils import class_weight
from scipy.special import logit, expit

class logit_norm:
    def __init__(self, array0, mean=True):
        array = np.copy(array0)
        self.shift = np.min(array, axis=0)
        self.num = len(self.shift)
        self.max = np.max(array, axis=0) - self.shift
        print(self.max)
        print(self.shift)
        if mean:
            finite=np.ones(len(array),dtype=bool)
            for i in range(self.num):
                array[:, i] = (array[:, i] - self.shift[i]) / self.max[i]
                print(np.max(array[:,i]),np.min(array[:,i]))
                array[:, i] = logit(array[:,i])
                finite *= np.isfinite(array[:, i])
            array=array[finite]
            print(sum(1-finite))
            self.mean = np.nanmean(array,axis=0)
            print(self.mean)
            self.std = np.nanstd(array,axis=0)
            print(self.std)
        self.do_mean = mean

    def forward(self, array0):
        array = np.copy(array0)
        finite = np.ones(len(array), dtype=bool)
        for i in range(self.num):
            array[:, i] = logit((array[:, i] - self.shift[i]) / self.max[i])
            if self.do_mean:
                array[:,i] = (array[:,i]-self.mean[i])/self.std[i]
            finite *= np.isfinite(array[:, i])
        return array[finite, :], finite

    def inverse(self, array0):
        array = np.copy(array0)
        for i in range(self.num):
            if self.do_mean:
                array[:,i] = array[:,i]*self.std[i] + self.mean[i]
            array[:,i] = expit(array[:,i])
            array[:, i] = array[:, i] * self.max[i] + self.shift[i]
        return array

def data(directory, m_central, inputs):
    #DATA Loading
    outerdata = np.load(directory+"data/outerdata.npy")[:, :inputs+1]
    print(outerdata.shape)
    norm = logit_norm(outerdata)

    outerdata_reduced = outerdata[outerdata[:,0]>2.8]
    number = (4*len(outerdata_reduced)//1000000 + 1)*1000000
    outersamples_reduced = sample.sample(outerdata_reduced, number, directory, True, m_central, norm)[:4*len(outerdata_reduced)]

    outerdata_train, outerdata_val = train_test_split(outerdata_reduced, test_size=0.2, random_state=42)
    outersamples_train, outersamples_val = train_test_split(outersamples_reduced, test_size=0.2, random_state=42)
    
    X_train = np.concatenate((outerdata_train, outersamples_train))
    X_val = np.concatenate((outerdata_val, outersamples_val))
    Y_train = np.append(np.ones(len(outerdata_train)), np.zeros(len(outersamples_train)))
    Y_val = np.append(np.ones(len(outerdata_val)), np.zeros(len(outersamples_val)))

    inds = np.arange(len(X_train))
    np.random.shuffle(inds)
    X_train = X_train[inds]
    print(X_train.shape)
    Y_train = Y_train[inds]
    print(sum(Y_train)/len(Y_train))

    inds = np.arange(len(X_val))
    np.random.shuffle(inds)
    X_val = X_val[inds]
    print(X_val.shape)
    Y_val = Y_val[inds]
    print(sum(Y_val)/len(Y_val))

    doctor_scaler = StandardScaler()
    doctor_scaler.fit(X_train)
    X_train = doctor_scaler.transform(X_train)
    X_val = doctor_scaler.transform(X_val)

    print(X_train.shape)
    print(X_val.shape)

    return X_train, Y_train, X_val, Y_val, doctor_scaler, norm

class Classifier(nn.Module):
    def __init__(self, layers, n_inputs=5):
        super().__init__()

        self.layers = []
        for nodes in layers:
            self.layers.append(nn.Linear(n_inputs, nodes))
            self.layers.append(nn.ReLU())
            n_inputs = nodes
        self.layers.append(nn.Linear(n_inputs, 1))
        self.layers.append(nn.Sigmoid())
        self.model_stack = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model_stack(x)

    def predict(self, x):
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        with torch.no_grad():
            self.eval()
            x = torch.tensor(x, device=device)
            prediction = self.forward(x).detach().cpu().numpy()
        return prediction


def build_classifier(filename, n_inputs=5):
    with open(filename, 'r') as stream:
        params = yaml.safe_load(stream)

    model = Classifier(params['layers'], n_inputs=n_inputs)
    if params['loss'] == 'binary_crossentropy':
        loss = F.binary_cross_entropy
    else:
        raise NotImplementedError

    if params['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=float(params['learning_rate']))
    else:
         raise NotImplementedError       

    return model, loss, optimizer


def train_model(classifier_configfile, epochs, X_train, Y_train, X_val, Y_val, direc_run, stop_after =10, batch_size=128, verbose=False):
    if not os.path.exists(direc_run):
        os.makedirs(direc_run)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(Y_train), y=Y_train)
    class_weights = dict(enumerate(class_weights))

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float), torch.tensor(Y_train, dtype=torch.float).reshape(-1,1))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float), torch.tensor(Y_val, dtype=torch.float).reshape(-1,1))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_loss = np.zeros(epochs)
    val_loss = np.zeros(epochs)

    model, loss_func, optimizer = build_classifier(classifier_configfile, n_inputs=X_train.shape[1])
    model.to(device)

    best_loss = 1000
    early_stopping = 0

    for epoch in range(epochs):
        print("training epoch nr", epoch)
        epoch_train_loss = 0.
        epoch_val_loss = 0.

        model.train()
        for i, batch in enumerate(train_dataloader):
            if verbose:
                print("...batch nr", i)
            batch_inputs, batch_labels = batch
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            batch_weights = (torch.ones(batch_labels.shape, device=device) - batch_labels)*class_weights[0] + batch_labels*class_weights[1]

            optimizer.zero_grad()
            batch_outputs = model(batch_inputs)
            batch_loss = loss_func(batch_outputs, batch_labels, weight=batch_weights)
            batch_loss.backward()
            optimizer.step()
            epoch_train_loss += batch_loss
            if verbose:
                print("...batch training loss:", batch_loss.item())

        epoch_train_loss /= (i+1)
        print("training loss:", epoch_train_loss.item())

        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(val_dataloader):
                batch_inputs, batch_labels = batch
                batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
                batch_weights = (torch.ones(batch_labels.shape, device=device)- batch_labels)*class_weights[0] + batch_labels*class_weights[1]

                batch_outputs = model(batch_inputs)
                batch_loss = loss_func(batch_outputs, batch_labels, weight=batch_weights)
                epoch_val_loss += batch_loss
            epoch_val_loss /= (i+1)
        print("validation loss:", epoch_val_loss.item(), flush=True)

        train_loss[epoch] = epoch_train_loss
        val_loss[epoch] = epoch_val_loss

        if epoch_val_loss < best_loss: 
            best_loss = epoch_val_loss
            torch.save(model, direc_run+"best_model.pt")
            early_stopping = 0
        else:
            early_stopping += 1
            if early_stopping > stop_after:
                break

    np.save(direc_run+"train_loss.npy", train_loss)
    np.save(direc_run+"val_loss.npy", val_loss)

    return torch.load(direc_run+"best_model.pt")