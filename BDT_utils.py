import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from pathlib import Path
from datetime import datetime

def make_one_array(twod_arr,new_arr):
	if len(new_arr) < len(twod_arr.T):
		app=np.zeros(len(twod_arr.T)-len(new_arr),dtype=None)
		new_arr=np.concatenate((new_arr,app),axis=0)
	elif len(twod_arr.T) < len(new_arr):
		app=np.zeros((len(twod_arr),len(new_arr)-len(twod_arr.T)),dtype=None)
		twod_arr=np.concatenate((twod_arr,app),axis=1)
	return np.concatenate((twod_arr,np.array([new_arr])),axis=0)

def plot_roc(test_results, test_labels, title="roc", directory = None, direc_run = None, save_AUC=False):
	if direc_run is None:
		direc_run=directory
		
	fpr, tpr, _ = roc_curve(test_labels, test_results)
	auc = roc_auc_score(test_labels, test_results)

	if Path(directory+"tpr_"+title+".npy").is_file():
		tpr_arr=np.load(directory+"tpr_"+title+".npy")
		np.save(directory+"tpr_"+title+".npy",make_one_array(tpr_arr,tpr))
		fpr_arr=np.load(directory+"fpr_"+title+".npy")
		np.save(directory+"fpr_"+title+".npy",make_one_array(fpr_arr,fpr))
	else: 
		np.save(directory+"tpr_"+title+".npy",np.array([tpr]))
		np.save(directory+"fpr_"+title+".npy",np.array([fpr]))

	if save_AUC:	
		f=open(directory+title+".txt",'a+')
		f.write("\n"+str(auc))
	return auc

def classifier_training(X_train, Y_train, X_test, Y_test, args, run):

    class_weight = {0: 1, 1: len(Y_train)/sum(Y_train.T[1])-1}
    class_weights = class_weight[0]*Y_train[:,0]+class_weight[1]*Y_train[:,1]

    print("\nTraining class weights: ", class_weight)

    test_results = np.zeros((args.ensemble_over,len(X_test)))

    for j in range(args.ensemble_over):
        print("Tree number:", args.ensemble_over*run+j)
        np.random.seed(int(datetime.now()))
        tree = HistGradientBoostingClassifier(verbose=1, max_iter=200, max_leaf_nodes=31, validation_fraction=0.5, random_state=int(datetime.now()))
        results_f = tree.fit(X_train, Y_train[:,1], sample_weight=class_weights)
        test_results[j] = tree.predict_proba(X_test)[:,1]
        #print("AUC last epoch: %.3f" % plot_roc(test_results[j], Y_test[:,1],title="roc_classifier",directory=args.directory))
        del tree
        del results_f
    test_results = np.mean(test_results, axis=0)
    #print("AUC last epoch: %.3f" % plot_roc(test_results, Y_test[:,1],title="roc_classifier_averaging",directory=args.directory))
