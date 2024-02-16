import numpy as np
import argparse
import os
from sklearn.ensemble import HistGradientBoostingClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
from pathlib import Path

parser = argparse.ArgumentParser(
    description='Run the full analysis chain.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#For SIC curve reproduction only these 5 option need to be changed
parser.add_argument('--bkg_file', type =str, required=True)
parser.add_argument('--samples_file', type=str, required=True)
parser.add_argument('--samples_weights', default=None, type=str)
parser.add_argument('--use_masses', default=False, action="store_true")
parser.add_argument('--directory', type=str, required=True)

parser.add_argument('--inputs', default=4, type=int)
parser.add_argument('--cl_logit', default=False, action="store_true")
parser.add_argument('--cl_norm', default=True, action="store_false")

#General classifier Arguments
parser.add_argument('--N_runs', type=int, default=10)
parser.add_argument('--ensemble_over', default=50, type=int)
parser.add_argument('--start_at_run', type=int, default=0)

args = parser.parse_args()

def classifier_training(X_train, Y_train, X_test, Y_test, args, run, weights):

    test_results = np.zeros((args.ensemble_over,len(X_test)))

    for j in range(args.ensemble_over):
        print("Tree number:", args.ensemble_over*run+j)
        np.random.seed(run*args.ensemble_over+j+1)
        tree = HistGradientBoostingClassifier(verbose=1, max_iter=200, max_leaf_nodes=31, validation_fraction=0.5, class_weight="balanced")
        results_f = tree.fit(X_train, Y_train, sample_weight=weights)
        test_results[j] = tree.predict_proba(X_test)[:,1]
        print("AUC last epoch: %.3f" % plot_roc(test_results[j], Y_test,title="roc_classifier",directory=args.directory, save_AUC=True))
        del tree
        del results_f
    test_results = np.mean(test_results, axis=0)
    print("AUC last epoch: %.3f" % plot_roc(test_results, Y_test,title="roc_classifier_averaging",directory=args.directory, save_AUC=True))

def make_one_array(twod_arr,new_arr):
	if len(new_arr) < len(twod_arr.T):
		app=np.zeros(len(twod_arr.T)-len(new_arr),dtype=None)
		new_arr=np.concatenate((new_arr,app),axis=0)
	elif len(twod_arr.T) < len(new_arr):
		app=np.zeros((len(twod_arr),len(new_arr)-len(twod_arr.T)),dtype=None)
		twod_arr=np.concatenate((twod_arr,app),axis=1)
	return np.concatenate((twod_arr,np.array([new_arr])),axis=0)

def plot_roc(test_results, test_labels, title="roc", directory = None, direc_run = None, plot=False, save_AUC=False):
	if direc_run is None:
		direc_run=directory
		
	fpr, tpr, _ = roc_curve(test_labels, test_results)
	auc = roc_auc_score(test_labels, test_results)

	if plot:
		x = np.linspace(0.001, 1, 10000)
		plt.figure()
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
			plt.plot(tpr, 1 / fpr, label="model")
		plt.plot(x, 1 / x, color="black", linestyle="--", label="random")
		plt.legend()
		plt.grid()
		plt.yscale("log")
		plt.ylim(1, 1e5)
		plt.xlim(0,1)
		plt.ylabel(r"1/$\epsilon_B$")
		plt.xlabel(r"$\epsilon_S$")
		plt.title(title)
		plt.savefig(direc_run+title+"roc.pdf")
		plt.close('all')

	if Path(directory+"tpr_"+title+".npy").is_file():
		tpr_arr=np.load(directory+"tpr_"+title+".npy")
		np.save(directory+"tpr_"+title+".npy",make_one_array(tpr_arr,tpr))
		fpr_arr=np.load(directory+"fpr_"+title+".npy")
		np.save(directory+"fpr_"+title+".npy",make_one_array(fpr_arr,fpr))
		if save_AUC:
			auc_arr = np.load(directory+"auc_"+title+".npy")
			auc_arr = np.append(auc_arr, auc)
			np.save(directory+"auc_"+title+".npy", auc_arr)
	else: 
		np.save(directory+"tpr_"+title+".npy",np.array([tpr]))
		np.save(directory+"fpr_"+title+".npy",np.array([fpr]))
		if save_AUC:
			np.save(directory+"auc_"+title+".npy", np.array([auc]))
	return auc

if not os.path.exists(args.directory):
	os.makedirs(args.directory)

samples = np.load(args.samples_file)
bkg = np.load(args.bkg_file)
if args.samples_weights is not None:
	samples_weights = np.load(args.samples_weights)

if args.use_masses:
	bkg = bkg[:,:args.inputs+1]
else:
	bkg = bkg[:, 1:args.inputs+1]
	samples = samples[:, 1:args.inputs+1]


bkg_train = bkg[:int(0.7*len(bkg))]
bkg_test = bkg[int(0.7*len(bkg)):]
samples_train = samples[:len(bkg_train)]
samples_test = samples[len(bkg_train):]

X_train = np.concatenate((bkg_train, samples_train), axis=0)
Y_train = np.concatenate((np.ones(len(bkg_train)),np.zeros(len(samples_train)) ), axis=0)
if args.samples_weights is not None:
	weights = np.concatenate((np.ones(len(bkg_train)), samples_weights[:len(bkg_train)]), axis=0)
else: 
	weights = np.ones(len(X_train))

X_test = np.concatenate((bkg_test, samples_test), axis=0)
Y_test = np.concatenate((np.ones(len(bkg_test)),np.zeros(len(samples_test)) ), axis=0)

for i in range(args.start_at_run, args.N_runs):
    print()
    print("------------------------------------------------------")
    print()
    print("Classifier run no. ", i)
    print()
    args.set_seed = i
    classifier_training(X_train, Y_train, X_test, Y_test, args, i, weights)