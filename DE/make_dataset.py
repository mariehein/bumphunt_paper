import numpy as np
import argparse
import os
import pandas as pd
import warnings

parser = argparse.ArgumentParser(
    description='Run the full analysis chain.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

#For SIC curve reproduction only these 5 option need to be changed
parser.add_argument('--directory', type=str, required=True)
parser.add_argument('--input_set', type=str, choices=["baseline","extended1","extended2","extended3","kitchensink", "kitchensink_super"])

#Data files
parser.add_argument('--data_file', type=str, default="/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_v2.extratau_2.features.h5")
parser.add_argument('--signal_file', type=str, default=None)
parser.add_argument('--three_pronged', default=False, action="store_true")

#Dataset preparation
parser.add_argument('--signal_percentage', type=float, default=None, help="Third in priority order")
parser.add_argument('--signal_number', type=int, default=None, help="Second in priority order")
parser.add_argument('--minmass', type=float, default=3.3)
parser.add_argument('--maxmass', type=float, default=3.7)
parser.add_argument('--window_number', type=int, default=None)
parser.add_argument('--set_seed', type=int, default=1)
parser.add_argument('--inputs', type=int, default=4)
parser.add_argument('--include_DeltaR', default=False, action="store_true")

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

if args.three_pronged:
	args.signal_file = "/hpcwork/rwth0934/LHCO_dataset/extratau2/events_anomalydetection_Z_XY_qqq.extratau_2.features.h5"
    
if args.window_number is not None:
    args.minmass = 3.3 - (5-args.window_number) * 0.1
    args.maxmass = 3.7 - (5-args.window_number) * 0.1

print(args)

def make_features_baseline(features, label_arr):
    E_part = np.sqrt(features[:,0]**2+features[:,1]**2+features[:,2]**2+features[:,3]**2)+np.sqrt(features[:,7]**2+features[:,8]**2+features[:,9]**2+features[:,10]**2)
    p_part2 = (features[:,0]+features[:,7])**2+(features[:,1]+features[:,8])**2+(features[:,2]+features[:,9])**2
    m_jj = np.sqrt(E_part**2-p_part2)
    ind=np.array(features[:,10]> features[:,3]).astype(int)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        feat1 = np.array([m_jj*1e-3, features[:, 3]*1e-3, (features[:,10]-features[:, 3])*1e-3, features[:, 5]/features[:,4], features[:, 12]/features[:,11], features[:, 6]/features[:,5], features[:, 13]/features[:,12], label_arr])
        feat2 = np.array([m_jj*1e-3, features[:, 10]*1e-3, (features[:,3]-features[:, 10])*1e-3, features[:, 12]/features[:,11], features[:, 5]/features[:,4], features[:, 13]/features[:,12], features[:, 6]/features[:,5], label_arr])
    return np.nan_to_num(feat1*ind+feat2*(np.ones(len(ind))-ind)).T

def make_features_extended12(features_j1, features_j2, label_arr, set):
	E_part2 = np.sqrt(features_j1[:,0]**2+features_j1[:,1]**2+features_j1[:,2]**2+features_j1[:,3]**2)+np.sqrt(features_j2[:,0]**2+features_j2[:,1]**2+features_j2[:,2]**2+features_j2[:,3]**2)
	p_part2 = (features_j1[:,0]+features_j2[:,0])**2 + (features_j1[:,1]+features_j2[:,1])**2 + (features_j1[:,2]+features_j2[:,2])**2
	m_jj = np.sqrt(E_part2**2-p_part2)

	ind_not = np.argwhere(features_j1[:,3]> features_j2[:,3])
	f_j1 = np.copy(features_j1)
	features_j1[ind_not] = features_j2[ind_not]
	features_j2[ind_not] = f_j1[ind_not]
	del f_j1	

	if set=="extended2":
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
			warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
			features = np.zeros((len(m_jj), 14))
			features[:,0] = m_jj * 1e-3
			features[:,1] = features_j1[:, 3] * 1e-3
			features[:,2] = (features_j2[:, 3] - features_j1[:, 3]) *1e-3
			for i in range(5):
				features[:,3+2*i] = features_j1[:,4+i]
				features[:,3+2*i+1] = features_j2[:,4+i]
			features[:,-1] = label_arr
	elif set=="extended1":
		with warnings.catch_warnings():
			warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
			warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
			inputs = 12
			features = np.zeros((len(m_jj), inputs))
			features[:,0] = m_jj * 1e-3
			features[:,1] = features_j1[:, 3] * 1e-3
			features[:,2] = (features_j2[:, 3] - features_j1[:, 3]) *1e-3
			for i in range(4):
				features[:,3+2*i] = features_j1[:,5+i]/features_j1[:,4+i]
				features[:,3+2*i+1] = features_j2[:,5+i]/features_j2[:,4+i]
			features[:,-1] = label_arr

	return np.nan_to_num(features)

def make_features_extended3(pandas_file, label_arr, set):
    features_j1 = np.array(pandas_file[['pxj1', 'pyj1', 'pzj1', 'mj1']], dtype=np.float32)
    features_j2 = np.array(pandas_file[['pxj2', 'pyj2', 'pzj2', 'mj2']], dtype=np.float32)
    
    E_part2 = np.sqrt(features_j1[:,0]**2+features_j1[:,1]**2+features_j1[:,2]**2+features_j1[:,3]**2)+np.sqrt(features_j2[:,0]**2+features_j2[:,1]**2+features_j2[:,2]**2+features_j2[:,3]**2)
    p_part2 = (features_j1[:,0]+features_j2[:,0])**2 + (features_j1[:,1]+features_j2[:,1])**2 + (features_j1[:,2]+features_j2[:,2])**2
    m_jj = np.sqrt(E_part2**2-p_part2)

    beta = [5, 1, 2]
    jet = ["j1", "j2"]
    to_subjettiness=9
    subjettinesses = np.zeros((len(m_jj),2,to_subjettiness*3))
    for k,b in enumerate(beta):
        for l, j in enumerate(jet):
            names = ["tau"+str(i)+j+"_"+str(b) for i in range(1,to_subjettiness+1)]
            subjettinesses[:,l,k*to_subjettiness:(k+1)*to_subjettiness]=np.array(pandas_file[names], dtype=np.float32)

    if set=="kitchensink":
        ratios = np.zeros((len(m_jj), 2, to_subjettiness-1))
        for l, j in enumerate(jet):
            names = ["tau"+str(i)+j+"_1" for i in range(1,to_subjettiness+1)]
            for i in range(len(names)-1):
                ratios[:,l, i]=np.array(pandas_file[names[i+1]], dtype=np.float32)/np.array(pandas_file[names[i]], dtype=np.float32)

    if set=="kitchensink_super":
        ratios = np.zeros((len(m_jj), 2, (to_subjettiness-1)*3))
        for k,b in enumerate(beta):
            for l, j in enumerate(jet):
                names = ["tau"+str(i)+j+"_"+str(b) for i in range(1,to_subjettiness+1)]
                for i in range(len(names)-1):
                    ratios[:,l, k*(to_subjettiness-1)+i]=np.array(pandas_file[names[i+1]], dtype=np.float32)/np.array(pandas_file[names[i]], dtype=np.float32)

    ind_not = np.argwhere(features_j1[:,3]> features_j2[:,3])
    f_j1 = np.copy(features_j1)
    features_j1[ind_not] = features_j2[ind_not]
    features_j2[ind_not] = f_j1[ind_not]
    del f_j1
    s = np.copy(subjettinesses)
    subjettinesses[ind_not,0] = subjettinesses[ind_not,1] 
    subjettinesses[ind_not,1] = s[ind_not,0] 
    del s
    if set in ["kitchensink", ["kitchensink_super"]]:
        r = np.copy(ratios)
        ratios[ind_not,0] = ratios[ind_not,1] 
        ratios[ind_not,1] = r[ind_not,0] 
        del r

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        features = np.zeros((len(m_jj), 4+to_subjettiness*6))
        if set=="kitchensink":
            features = np.zeros((len(m_jj), 4+to_subjettiness*8-2))
        elif set=="kitchensink_super":
            features = np.zeros((len(m_jj), 4+to_subjettiness*6+(to_subjettiness-1)*6))
        features[:,0] = m_jj * 1e-3
        features[:,1] = features_j1[:, 3] * 1e-3
        features[:,2] = (features_j2[:, 3] - features_j1[:, 3]) *1e-3
        features[:,3:3+subjettinesses.shape[-1]] = subjettinesses[:,0]
        features[:,3+subjettinesses.shape[-1]:3+subjettinesses.shape[-1]*2]= subjettinesses[:,1]
        if set in ["kitchensink", "kitchensink_super"]:
            features[:,3+subjettinesses.shape[-1]*2:3+subjettinesses.shape[-1]*2+ratios.shape[-1]] = ratios[:,0] 
            features[:,3+subjettinesses.shape[-1]*2+ratios.shape[-1] : 3+subjettinesses.shape[-1]*2+ratios.shape[-1]*2] = ratios[:,1] 
        features[:,-1] = label_arr

    return np.nan_to_num(features)

def file_loading(filename, args, labels=True, signal=0):
    pandas_file = pd.read_hdf(filename)
    if labels:
        label_arr = np.array(pandas_file['label'], dtype=float)
    else: 
        label_arr = np.ones((len(pandas_file['pxj1'])), dtype=float)*signal

    if args.input_set == "baseline":
        features = np.array(pandas_file[['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1_1', 'tau2j1_1', 'tau3j1_1', 'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2_1', 'tau2j2_1', 'tau3j2_1']], dtype=float)
        features = make_features_baseline(features, label_arr)
    elif args.input_set in ["extended1", "extended2"]:
        features_j1 = np.array(pandas_file[['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1_1', 'tau2j1_1', 'tau3j1_1', 'tau4j1_1', 'tau5j1_1']], dtype=float)
        features_j2 = np.array(pandas_file[['pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2_1', 'tau2j2_1', 'tau3j2_1', 'tau4j2_1', 'tau5j2_1']], dtype=float)
        features = make_features_extended12(features_j1, features_j2, label_arr, args.input_set)
        del features_j1, features_j2
    elif args.input_set in ["extended3","kitchensink", "kitchensink_super"]:
        features = make_features_extended3(pandas_file, label_arr, args.input_set)
    del pandas_file
    return features

def DR(filename, labels=True):
	if labels:
		features = np.array(pd.read_hdf(filename)[['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1_1', 'tau2j1_1', 'tau3j1_1', 'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2_1', 'tau2j2_1', 'tau3j2_1']],dtype=float)
	else: 
		features = np.array(pd.read_hdf(filename)[['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1_1', 'tau2j1_1', 'tau3j1_1', 'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2_1', 'tau2j2_1', 'tau3j2_1']],dtype=float)
		features = np.concatenate((features,np.zeros((len(features),1))),axis=1)

	phi1 = np.arctan2(features[:,1], features[:,0])
	phi2 = np.arctan2(features[:,8], features[:,7])
	Dphi = np.abs(phi2-phi1)
	Dphi = Dphi - np.where(Dphi>np.pi*2, 2*np.pi,0)
	eta1 = np.arcsinh(features[:,2]/np.sqrt(features[:,1]**2+ features[:,0]**2))
	eta2 = np.arcsinh(features[:,9]/np.sqrt(features[:,7]**2+ features[:,8]**2))
	DR = np.sqrt((Dphi)**2 + (eta1-eta2)**2)

	return DR

def make_DE_data(args):
    data = file_loading(args.data_file, args)
    if args.signal_file is not None: 
        data_signal = file_loading(args.signal_file, args, labels=False, signal=1)

    if args.signal_file is not None: 
        sig = data_signal
    else:
        sig = data[data[:,-1]==1]

    if args.include_DeltaR:
        data_DR = DR(args.data_file)
        data = np.concatenate((data[:,:args.inputs],np.array([data_DR]).T, data[:,args.inputs:]),axis=1)
        if args.signal_file is not None:
            sig_DR = DR(args.signal_file, labels=False)
            sig = np.concatenate((sig[:,:args.inputs],np.array([sig_DR]).T, sig[:,args.inputs:]),axis=1)
        else:
            sig = data[data[:,-1]==1]
    bkg = data[data[:,-1]==0]
    print(len(bkg), len(sig))

    if args.signal_number is not None:
        n_sig=args.signal_number
    elif args.signal_percentage is None:
        n_sig = 1000
    else:
        n_sig = int(args.signal_percentage*1000/0.6361658645922605)
    print("n_sig=", n_sig)

    data_all = np.concatenate((bkg,sig[:n_sig]),axis=0)
    np.random.seed(args.set_seed)
    np.random.shuffle(data_all)

    innermask = (data_all[:,0]>args.minmass) & (data_all[:,0]<args.maxmass)
    innerdata = data_all[innermask]
    outerdata = data_all[~innermask]
    np.save(args.directory+"innerdata.npy",innerdata)
    np.save(args.directory+"outerdata.npy",outerdata)


make_DE_data(args)