import numpy as np
import kfold_utils as dp
import argparse
import os
import BDT_utils as BDT

parser = argparse.ArgumentParser(
    description='Run the full analysis chain.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#For SIC curve reproduction only these 5 option need to be changed
parser.add_argument('--mode', type=str, choices=["IAD", "cwola", "cathode", "IAD_scan"], required=True)
parser.add_argument('--fold_number', type=int, required=True)
parser.add_argument('--window_number', type=int, required=True)
parser.add_argument('--directory', type=str, required=True)
parser.add_argument('--input_set', type=str, choices=["baseline","extended1","extended2","extended3"], required=True)

#Data files
parser.add_argument('--data_file', type=str, default="/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_v2.extratau_2.features.h5")
parser.add_argument('--extrabkg_file', type=str, default="/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_DelphesPythia8_v2_qcd_extra_inneronly_combined_extratau_2_features.h5")
parser.add_argument('--signal_file', type=str, default=None)
parser.add_argument('--three_pronged', default=False, action="store_true")
parser.add_argument('--samples_file', default=None, type=str)
parser.add_argument('--samples_file_array', default=False, action="store_true")
parser.add_argument('--samples_file_start', default=None, type=str)
parser.add_argument('--samples_file_ending', default=None, type=str)
parser.add_argument('--samples_ranit', default=False, action="store_true")
parser.add_argument('--samples_weights', default=None, type=str)

#Dataset preparation
parser.add_argument('--signal_percentage', type=float, default=None, help="Second in priority order")
parser.add_argument('--signal_number', type=int, default=None, help="Top priority")
parser.add_argument('--fixed_oversampling', type=int, default=4)
parser.add_argument('--N_train', type=int, default=320000)
parser.add_argument('--minmass', type=float, default=3.3)#modified based on window number
parser.add_argument('--maxmass', type=float, default=3.7)#modified based on window number
parser.add_argument('--ssb_width', type=float, default=0.2)
parser.add_argument('--cl_norm', default=True, action="store_false")
parser.add_argument('--set_seed', type=int, default=1)
parser.add_argument('--inputs', type=int, default=4)
parser.add_argument('--inputs_custom', type=int, default=None)
parser.add_argument('--randomize_seed', default=False, action="store_true")
parser.add_argument('--include_DeltaR', default=False, action="store_true")
parser.add_argument('--Herwig', default=False, action="store_true")
parser.add_argument('--Joep', default=False, action="store_true")


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

if args.include_DeltaR:
    args.inputs+=1

if args.inputs_custom is not None:
    args.inputs = args.inputs_custom


if args.three_pronged:
	args.signal_file = "/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_Z_XY_qqq.extratau_2.features.h5"
     
if args.Herwig:
    args.data_file = "/hpcwork/rwth0934/LHCO_dataset/Herwig/events_anomalydetection_herwig.extratau_2.features.h5"
if args.Joep:
    args.data_file = "/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_joep.extratau_2.features.h5"

args.minmass = (args.window_number-5)*0.1+3.3
args.maxmass = (args.window_number-5)*0.1+3.7

if args.mode=="IAD" and args.window_number!=5:
    raise ValueError("IAD currently only supported with window 5, choose IAD_scan to use less data but all windows")

print(args)

if not args.randomize_seed and not args.samples_file_array:
    X_train, Y_train, X_test, Y_test, samples_test, train_weights = dp.k_fold_data_prep(args)
for i in range(args.start_at_run, args.N_runs):
    print()
    print("------------------------------------------------------")
    print()
    print("Classifier run no. ", i)
    print()
    if args.randomize_seed:
        args.set_seed = i
        X_train, Y_train, X_test, Y_test, samples_test, train_weights = dp.k_fold_data_prep(args)
    if args.samples_file_array: 
        args.samples_file = args.samples_file_start + str(i+1) + args.samples_file_ending
        X_train, Y_train, X_test, Y_test, samples_test, train_weights = dp.k_fold_data_prep(args)
    BDT.classifier_training(X_train, Y_train, X_test, samples_test, train_weights, args, i)
