import numpy as np
import dataprep_utils as dp
import plotting_utils as pf
import argparse
import os
import BDT_utils as BDT

parser = argparse.ArgumentParser(
    description='Run the full analysis chain.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#For SIC curve reproduction only these 5 option need to be changed
parser.add_argument('--mode', type=str, choices=["IAD", "supervised"], required=True)
parser.add_argument('--directory', type=str, required=True)
parser.add_argument('--input_set', type=str, choices=["baseline","extended1","extended2","extended3","kitchensink", "kitchensink_super"])
parser.add_argument('--gaussian_inputs', type=int, default=None)

#Data files
parser.add_argument('--data_file', type=str, default="/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_v2.extratau_2.features.h5")
parser.add_argument('--extrabkg_file', type=str, default="/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_DelphesPythia8_v2_qcd_extra_inneronly_combined_extratau_2_features.h5")
parser.add_argument('--signal_file', type=str, default=None)
parser.add_argument('--samples_file', type=str, default=None)
parser.add_argument('--samples_ranit', default=False, action="store_true")
parser.add_argument('--samples_weights', type=str, default=None)
parser.add_argument('--three_pronged', default=False, action="store_true")

#Dataset preparation
parser.add_argument('--signal_percentage', type=float, default=None, help="Third in priority order")
parser.add_argument('--signal_number', type=int, default=None, help="Second in priority order")
parser.add_argument('--minmass', type=float, default=3.3)
parser.add_argument('--maxmass', type=float, default=3.7)
parser.add_argument('--cl_logit', default=False, action="store_true")
parser.add_argument('--cl_norm', default=True, action="store_false")
parser.add_argument('--N_normal_inputs', default=4, type=int, help="Needed only for gaussian inputs")
parser.add_argument('--supervised_normal_signal', default=False, action="store_true")
parser.add_argument('--set_seed', type=int, default=1)
parser.add_argument('--randomize_seed', default=False, action="store_true")
parser.add_argument('--sample_test', default=False, action="store_true")
parser.add_argument('--inputs', type=int, default=4)
parser.add_argument('--include_DeltaR', default=False, action="store_true")
parser.add_argument('--reduced_stats', default=False, action="store_true")

#2D scan
parser.add_argument('--scan_2D', default=False, action="store_true")
parser.add_argument('--N_CR', type=int, default=None)
parser.add_argument('--N_bkg', type=int, default=None)
parser.add_argument('--signal_significance', type=float, default=None, help="Takes highest priority, only supported for 2D_scan")

#General classifier Arguments
parser.add_argument('--N_runs', type=int, default=10)
parser.add_argument('--ensemble_over', default=50, type=int)
parser.add_argument('--start_at_run', type=int, default=0)

args = parser.parse_args()

if not os.path.exists(args.directory):
	os.makedirs(args.directory)

if args.input_set=="extended1":
    args.inputs=10
elif args.input_set=="extended2":
    args.inputs=12
elif args.input_set=="extended3":
    args.inputs=56
elif args.input_set=="kitchensink":
    args.inputs=72
elif args.input_set=="kitchensink_super":
    args.inputs=104

if args.include_DeltaR:
    args.inputs+=1

if args.sample_test:
    args.signal_number=0

if args.three_pronged:
	args.signal_file = "/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_Z_XY_qqq.extratau_2.features.h5"

if args.gaussian_inputs is not None:
    args.inputs+=args.gaussian_inputs

if args.scan_2D: 
    if args.N_CR is None:
        args.N_CR = args.N_bkg
    if not args.signal_significance:
        raise ValueError("need signal significance for 2D_scan")

if args.signal_significance is not None:
    if not args.scan_2D: 
        raise ValueError("signal significance only supported for 2D_scan")
    
print(args)

if not args.scan_2D:
    if not args.randomize_seed:
        X_train, Y_train, train_weights, X_test, Y_test = dp.classifier_data_prep(args)
        if args.reduced_stats:
            train_len = int(len(X_train)*0.8)
            X_train = X_train[:train_len]
            Y_train = Y_train[:train_len]
            train_weights = train_weights[:train_len]
    for i in range(args.start_at_run, args.N_runs):
        print()
        print("------------------------------------------------------")
        print()
        print("Classifier run no. ", i)
        print()
        args.set_seed = i
        if args.randomize_seed:
            X_train, Y_train, train_weights, X_test, Y_test = dp.classifier_data_prep(args)
            if args.reduced_stats:
                train_len = int(len(X_train)*0.8)
                X_train = X_train[:train_len]
                Y_train = Y_train[:train_len]
                train_weights = train_weights[:train_len]
        BDT.classifier_training(X_train, Y_train, train_weights, X_test, Y_test, args, i)

else: 
    if args.samples_weights is not None:
        raise ValueError("Currently 2D scan does not support samples_weights")
    if args.N_bkg is None:
        raise ValueError("Need to specify N_bkg")
    if args.classifier=="NN":
        raise ValueError("NN 2D scan currently not supported.")
    else:
        for i in range(args.start_at_run, args.N_runs):
            print()
            print("------------------------------------------------------")
            print()
            print("Classifier run no. ", i)
            print()
            args.set_seed = i
            if args.randomize_seed:
                X_train, Y_train, X_test, Y_test = dp.data_prep_2D(args)
            BDT.classifier_training(X_train, Y_train, X_test, Y_test, args, i)
