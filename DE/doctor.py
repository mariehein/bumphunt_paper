import numpy as np
import argparse
import doctor_utils as ute
import torch
from neural_network_classifier import NeuralNetworkClassifier

parser = argparse.ArgumentParser(
    description='Run the full analysis chain.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#For SIC curve reproduction only these 5 option need to be changed
parser.add_argument('--directory', type=str, required=True)
parser.add_argument('--save_directory', type=str, default=None)
parser.add_argument('--m_central', type=float, default=3.5)
parser.add_argument('--weight_sidebands', default=False, action="store_true")
parser.add_argument('--inputs', type=int, default=4)
args = parser.parse_args()

if args.save_directory is None:
    args.save_directory = args.directory+"reweighting/"

def weight_SBs(X):
    left = X[:,0]<args.m_central
    right = X[:,0]>args.m_central
    N_left = sum(left)
    N_right = sum(right)
    weights = np.ones(len(X))
    weights[left]= 2*N_right/len(X)
    weights[right]= 2*N_left/len(X)
    weights.reshape((len(weights)))
    return weights

X_train, Y_train, X_val, Y_val, doctor_scaler, norm = ute.data(args.directory, args.m_central, args.inputs)
if args.weight_sidebands:
    train_weights = weight_SBs(X_train)
    val_weights = weight_SBs(X_val)
else: 
    train_weights = None
    val_weights = None

doctor_classifier_model = NeuralNetworkClassifier(save_path=args.save_directory,
                                                  n_inputs=X_train.shape[1],
                                                  lr=1e-4,
                                                  early_stopping=True,
                                                  epochs=None,
                                                  verbose=False)
doctor_classifier_model.fit(X_train, Y_train,X_val, Y_val, sample_weight = train_weights, sample_weight_val = val_weights)
doctor_classifier_model.load_best_model()

#classifier = ute.train_model("doctor.yml", 100, X_train, Y_train, X_val, Y_val, args.directory+"reweighting_Manuel/")

innerdata = np.load(args.directory+"data/innerdata.npy")[:,:args.inputs+1]
samples = np.array(ute.sample.sample(innerdata, 50000000, args.directory, False, args.m_central, norm), dtype=np.float32)
samples_loader = torch.utils.data.DataLoader(samples, batch_size=1000, shuffle=False)
weights = np.zeros(len(samples))
for i, batch in enumerate(samples_loader):
    p = doctor_classifier_model.predict(np.array(doctor_scaler.transform(batch), dtype=np.float32))[:,0]
    weights[i*1000:(i+1)*1000] = p/(1-p)
samples = samples[np.random.choice(len(samples), 1000000, p=weights/np.sum(weights))]

np.save(args.save_directory+"samples_reweighted.npy", samples)
