import numpy as np
#import dataprep_utils as dp
import MAF_utils as MAF
import argparse
import os
from scipy.special import logit, expit


parser = argparse.ArgumentParser(
    description='Run the full analysis chain.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_set', type=str, choices=["baseline","extended1","extended2","extended3"])
parser.add_argument('--directory', type=str, required=True)
parser.add_argument('--inputs', type=int, default=4)
parser.add_argument('--conditional_inputs', type=int, default=1)
parser.add_argument('--N_samples', type=int, default=1000000)
parser.add_argument('--DE_filename', type=str, default="MAF.yml")
parser.add_argument('--data_direc', type=str, default = "data/baseline_without/")

#parser.add_argument('--DE_N_best_epochs', type=int, default=10)
#parser.add_argument('--no_averaging', default=True, action="store_false")
#parser.add_argument('--weight_averaging', default=True, action="store_false")
#parser.add_argument('--ensemble', default=True, action="store_false")

parser.add_argument("--batch_size", default = 128, type=int)
parser.add_argument("--epochs", default = 100, type=int)
parser.add_argument("--patience_max", default = 10, type=int)
parser.add_argument("--hidden", default = 128, type=int)
parser.add_argument("--transforms", default = 20, type=int)
parser.add_argument("--blocks", default = 1, type=int)
parser.add_argument("--learning_rate", default = 1e-3, type=float)
parser.add_argument('--save_models', default=False, action="store_true")

args = parser.parse_args()
print(args)

if not os.path.exists(args.directory):
	os.makedirs(args.directory)


np.save(args.directory+"args.npy", args)

if args.input_set=="extended1":
    args.inputs=10
elif args.input_set=="extended2":
    args.inputs=12
elif args.input_set=="extended3":
    args.inputs=56


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

data = np.load(args.data_direc+"outerdata.npy")[:,:args.inputs+1]
print(data.shape)
inner = np.load(args.data_direc+"innerdata.npy")[:,:args.inputs+1]
print(inner.shape)
train_val_test=np.array([0.6, 0.8])
N = len(data)

train_val_test = np.array(train_val_test*N, dtype=int)
print(train_val_test)
train_data = data[:train_val_test[0]]
val_data = data[train_val_test[0]:train_val_test[1]]
test_data = data[train_val_test[1]:]
np.save(args.directory+"test_data.npy", test_data)

norm = logit_norm(train_data, mean=True)#, no_logit=1)
train_data, _ = norm.forward(train_data)
val_data, _ = norm.forward(val_data)
inner_normed, _ = norm.forward(inner)
m_inner = inner_normed[:,0]
m_outer = train_data[:,0]

MAF.run_MAF(args, train_data, val_data, m_inner, m_outer, test_data, inner, norm)
