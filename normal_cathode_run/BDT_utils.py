import numpy as np
import plotting_utils as pf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss



def classifier_training(X_train, Y_train, train_weights, X_test, Y_test, args, run):

    test_results = np.zeros((args.ensemble_over,len(X_test)))

    for j in range(args.ensemble_over):
        print("Tree number:", args.ensemble_over*run+j)
        np.random.seed(run*args.ensemble_over+j+1)
        tree = HistGradientBoostingClassifier(verbose=1, max_iter=200, max_leaf_nodes=31, validation_fraction=0.5, class_weight="balanced")
        results_f = tree.fit(X_train, Y_train, sample_weight=train_weights)
        test_results[j] = tree.predict_proba(X_test)[:,1]
        print("AUC last epoch: %.3f" % pf.plot_roc(test_results[j], Y_test,title="roc_classifier",directory=args.directory, save_AUC=args.sample_test))
        del tree
        del results_f
    test_results = np.mean(test_results, axis=0)
    np.save(args.directory+"preds.npy", test_results)
    np.save(args.directory+"labels.npy", Y_test)
    print("AUC last epoch: %.3f" % pf.plot_roc(test_results, Y_test,title="roc_classifier_averaging",directory=args.directory, save_AUC=args.sample_test))
