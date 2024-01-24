import numpy as np
import kfold_utils as dp
import argparse
import os
import BDT_utils as BDT

parser = argparse.ArgumentParser(
    description='Run the full analysis chain.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#For SIC curve reproduction only these 5 option need to be changed
parser.add_argument('--mode', type=str, choices=["IAD", "cwola", "cathode"], required=True)
parser.add_argument('--fold_number', type=int, required=True)
parser.add_argument('--window_number', type=int, required=True)
parser.add_argument('--directory', type=str, required=True)
parser.add_argument('--input_set', type=str, choices=["baseline","extended1","extended2","extended3"])

#Data files
parser.add_argument('--data_file', type=str, default="/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_v2.extratau_2.features.h5")
parser.add_argument('--extrabkg_file', type=str, default="/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_DelphesPythia8_v2_qcd_extra_inneronly_combined_extratau_2_features.h5")
parser.add_argument('--signal_file', type=str, default=None)
parser.add_argument('--three_pronged', default=False, action="store_true")
parser.add_argument('--sample_file', default=None, type=str)

#Dataset preparation
parser.add_argument('--signal_percentage', type=float, default=None, help="Second in priority order")
parser.add_argument('--signal_number', type=int, default=None, help="Top priority")
parser.add_argument('--N_train', type=int, default=320000)
parser.add_argument('--minmass', type=float, default=3.3)#modified based on window number
parser.add_argument('--maxmass', type=float, default=3.7)#modified based on window number
parser.add_argument('--cl_norm', default=True, action="store_false")
parser.add_argument('--set_seed', type=int, default=1)
parser.add_argument('--inputs', type=int, default=4)

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

if args.three_pronged:
	args.signal_file = "/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_Z_XY_qqq.extratau_2.features.h5"

args.minmass = (args.window_number-5)*0.1+3.3
args.minmass = (args.window_number-5)*0.1+3.7

if args.mode=="IAD" and args.window_number!=5:
    raise ValueError("IAD currently only supported with window 5")

print(args)

X_train, Y_train, X_test, Y_test = dp.classifier_data_prep(args)
for i in range(args.start_at_run, args.N_runs):
    print()
    print("------------------------------------------------------")
    print()
    print("Classifier run no. ", i)
    print()
    BDT.classifier_training(X_train, Y_train, X_test, Y_test, args)
