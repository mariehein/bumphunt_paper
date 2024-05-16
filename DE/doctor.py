import numpy as np
import argparse
import doctor_utils as ute
import torch

parser = argparse.ArgumentParser(
    description='Run the full analysis chain.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#For SIC curve reproduction only these 5 option need to be changed
parser.add_argument('--directory', type=str, required=True)
parser.add_argument('--m_central', type=float, default=3.5)
parser.add_argument('--inputs', type=int, default=4)
args = parser.parse_args()

X_train, Y_train, X_val, Y_val, doctor_scaler, norm = ute.data(args.directory, args.m_central, args.inputs)
classifier = ute.train_model("doctor.yml", 100, X_train, Y_train, X_val, Y_val, args.directory+"reweighting_Manuel/")

innerdata = np.load(args.directory+"data/innerdata.npy")[:,:args.inputs+1]
samples = np.array(ute.sample.sample(innerdata, 50000000, args.directory, False, args.m_central, norm), dtype=np.float32)
samples_loader = torch.utils.data.DataLoader(samples, batch_size=1000, shuffle=False)
weights = np.zeros(len(samples))
for i, batch in enumerate(samples_loader):
    p = classifier.predict(np.array(doctor_scaler.transform(batch), dtype=np.float32))[:,0]
    weights[i*1000:(i+1)*1000] = p/(1-p)
samples = samples[np.random.choice(len(samples), 1000000, p=weights/np.sum(weights))]

np.save(args.directory+"samples_reweighted.npy", samples)
