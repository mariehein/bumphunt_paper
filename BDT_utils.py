import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from pathlib import Path
from datetime import datetime

def add_to_file(file, arr):
	if Path(file).is_file():
		np.save(file, np.concatenate((np.load(file), np.reshape(arr, (1,len(arr))))))
	else:
		np.save(file, np.reshape(arr, (1,len(arr))))

def classifier_training(X_train, Y_train, X_test, samples_test, args, run):
	test_results = np.zeros((args.ensemble_over,len(X_test)))
	samples_results = np.zeros((args.ensemble_over, len(samples_test)))

	for j in range(args.ensemble_over):
		print("Tree number:", args.ensemble_over*run+j)
		np.random.seed(int(datetime.now().timestamp()))
		tree = HistGradientBoostingClassifier(verbose=0, max_iter=200, max_leaf_nodes=31, validation_fraction=0.5, random_state=int(datetime.now().timestamp()), class_weight="balanced")
		results_f = tree.fit(X_train, Y_train)
		test_results[j] = tree.predict_proba(X_test)[:,1]
		samples_results[j] = tree.predict_proba(samples_test)[:,1]
		del tree
		del results_f
	test_results = np.mean(test_results, axis=0)
	samples_results = np.mean(samples_results, axis=0)

	add_to_file(args.directory+"test_preds.npy", test_results)
	add_to_file(args.directory+"samples_preds.npy", samples_results)
	
