import numpy as np
import kfold_utils as dp
import argparse
import os
import BDT_utils as BDT

parser = argparse.ArgumentParser(
    description='Run the full analysis chain.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#For SIC curve reproduction only these 5 option need to be changed
parser.add_argument('--mode', type=str, choices=["IAD", "cwola", "cathode", "IAD_scan", "IAD_joep"], required=True, help="Choose background template mode")
parser.add_argument('--fold_number', type=int, required=True, help="fold number for k-fold cross validation, should be between 0 and 4")
parser.add_argument('--window_number', type=int, required=True, help="window number used to specify minmass and maxmass, between 1 and 9")
parser.add_argument('--directory', type=str, required=True, help="save directory")
parser.add_argument('--input_set', type=str, choices=["baseline","extended1","extended2","extended3"], required=True, help="input feature set")
parser.add_argument('--include_DeltaR', default=False, action="store_true", help="appending Delta R to input feature set")
parser.add_argument('--signal_number', type=int, default=None, help="Number of signal events in data set")

#Data files
parser.add_argument('--data_file', type=str, default="/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_v2.extratau_2.features.h5")
parser.add_argument('--extrabkg_file', type=str, default="/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_DelphesPythia8_v2_qcd_extra_inneronly_combined_extratau_2_features.h5")
parser.add_argument('--signal_file', type=str, default=None, help="Specify different signal file")
parser.add_argument('--three_pronged', default=False, action="store_true", help="Activate three-pronged signal file")
parser.add_argument('--samples_file', default=None, type=str, help="Samples file for cathode, assumed as .npy")
parser.add_argument('--samples_file_array', default=False, action="store_true", help="Independent samples file per classiifer")
parser.add_argument('--samples_file_start', default=None, type=str, help="start of samples file if --sample_file_array")
parser.add_argument('--samples_file_ending', default=None, type=str, help="end of samples file if --sample_file_array")
parser.add_argument('--samples_ranit', default=False, action="store_true", help="Activate if samples contain labels")
parser.add_argument('--samples_weights', default=None, type=str, help="reweight samples")
parser.add_argument('--extra_signal_predictions', default=False, action="store_true")

#Dataset preparation
parser.add_argument('--signal_percentage', type=float, default=None, help="alternate way of specifying signal number, second priority")
parser.add_argument('--fixed_oversampling', type=int, default=4, help="Oversampling factor for cathode")
parser.add_argument('--N_train', type=int, default=320000, help="Oversampling specified as total event number")
parser.add_argument('--minmass', type=float, default=3.3, help="overwritten by window number")
parser.add_argument('--maxmass', type=float, default=3.7, help="overwritten by window number")
parser.add_argument('--ssb_width', type=float, default=0.2, help="SSB width for CWoLa")
parser.add_argument('--cl_norm', default=True, action="store_false", help="Norming data for classification")
parser.add_argument('--set_seed', type=int, default=1, help="Seed for data set preparation, does not affect signal events unless --randomize_signal is not None")
parser.add_argument('--inputs', type=int, default=4, help="specified internally")
parser.add_argument('--inputs_custom', type=int, default=None, help="specified internally")
parser.add_argument('--randomize_seed', default=False, action="store_true", help="for dataset randomization")
parser.add_argument('--randomize_signal', default=None, type=int, help="for signal event randomization")

#Select data on which to run
parser.add_argument('--Herwig', default=False, action="store_true", help="MC runs")
parser.add_argument('--Joep', default=False, action="store_true", help="reproduction of LHCO dataset")
parser.add_argument('--cathode_on_SBs', default=False, action="store_true", help="data-driven estimate of deltasys for cathode")
parser.add_argument('--cathode_on_SSBs', default=False, action="store_true", help="data-driven estimate of deltasys using only SSB")
parser.add_argument('--select_SB_data', default=True, action='store_false', help="subsampling of SB data for data-driven estimate of deltasys")


#General classifier Arguments
parser.add_argument('--N_runs', type=int, default=10, help="number of independent classifier runs")
parser.add_argument('--ensemble_over', default=50, type=int, help="number of ensembled classifiers")
parser.add_argument('--start_at_run', type=int, default=0, help="allows for warm restart between independent classifier runs")

args = parser.parse_args()

# Change options if dependent on one another
if args.cathode_on_SSBs:
    args.cathode_on_SBs = True

if not os.path.exists(args.directory):
	os.makedirs(args.directory)

if args.input_set=="extended1":
    args.inputs=10
elif args.input_set=="extended2":
    args.inputs=12
elif args.input_set=="extended3":
    args.inputs=56

if args.include_DeltaR:
    args.inputs+=1

if args.inputs_custom is not None:
    args.inputs = args.inputs_custom

if args.mode=="IAD_joep":
    args.Joep=True

if args.three_pronged:
	args.signal_file = "/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_Z_XY_qqq.extratau_2.features.h5"
     
if args.Herwig:
    args.data_file = "/hpcwork/rwth0934/LHCO_dataset/Herwig/events_anomalydetection_herwig.extratau_2.features.h5"
if args.Joep:
    args.data_file = "/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_joep_all.extratau_2.features.h5"
    args.extrabkg_file = "/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_joep_extrabkg.extratau_2.features.h5"

args.minmass = (args.window_number-5)*0.1+3.3
args.maxmass = (args.window_number-5)*0.1+3.7

if args.mode=="IAD" and args.window_number!=5:
    raise ValueError("IAD currently only supported with window 5, choose IAD_scan to use less data but all windows")

print(args)

# make training set and run classifiers
if not args.randomize_seed and not args.randomize_signal and not args.samples_file_array:
    X_train, Y_train, X_test, Y_test, samples_test, signal_test, train_weights = dp.k_fold_data_prep(args)
for i in range(args.start_at_run, args.N_runs):
    print()
    print("------------------------------------------------------")
    print()
    print("Classifier run no. ", i)
    print()
    if args.randomize_seed or args.randomize_signal is not None and not args.samples_file_array:
        args.set_seed = i
        if args.randomize_signal is not None: 
            args.randomize_signal = i
        X_train, Y_train, X_test, Y_test, samples_test, signal_test, train_weights = dp.k_fold_data_prep(args)
    if args.samples_file_array: 
        if args.randomize_seed:
            args.set_seed = i
        if args.randomize_signal is not None: 
            args.randomize_signal = i
        if args.samples_ranit:
            args.samples_file = args.samples_file_start + str(i) + args.samples_file_ending
        else:
            args.samples_file = args.samples_file_start + str(i+1) + args.samples_file_ending
        X_train, Y_train, X_test, Y_test, samples_test, signal_test, train_weights = dp.k_fold_data_prep(args)
    BDT.classifier_training(X_train, Y_train, X_test, Y_test, samples_test, signal_test, train_weights, args, i)
