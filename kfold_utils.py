import numpy as np
import pandas as pd
import warnings
import os

def to_categorical(Y, N_classes):
	Y=np.array(Y,dtype=int)
	return np.eye(N_classes)[Y]

def shuffle_XY(X,Y):
    seed_int=np.random.randint(300)
    np.random.seed(seed_int)
    np.random.shuffle(X)
    np.random.seed(seed_int)
    np.random.shuffle(Y)
    return X,Y

class no_logit_norm:
	def __init__(self,array):
		self.mean = np.mean(array, axis=0)
		self.std = np.std(array, axis=0)

	def forward(self,array0):
		return (np.copy(array0)-self.mean)/self.std, np.ones(len(array0),dtype=bool)

	def inverse(self,array0):
		return np.copy(array0)*self.std+self.mean

def file_loading(filename, labels=True):
	if labels:
		features = np.array(pd.read_hdf(filename)[['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1', 'tau2j1', 'tau3j1', 'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2', 'tau2j2', 'tau3j2', 'label']])
	else: 
		features = np.array(pd.read_hdf(filename)[['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1', 'tau2j1', 'tau3j1', 'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2', 'tau2j2', 'tau3j2']])
		features = np.concatenate((features,np.zeros((len(features),1))),axis=1)
	E_part = np.sqrt(features[:,0]**2+features[:,1]**2+features[:,2]**2+features[:,3]**2)+np.sqrt(features[:,7]**2+features[:,8]**2+features[:,9]**2+features[:,10]**2)
	p_part = (features[:,0]+features[:,7])**2+(features[:,1]+features[:,8])**2+(features[:,2]+features[:,9])**2
	m_jj = np.sqrt(E_part**2-p_part)
	#p_t = np.array([np.max([np.sqrt(features[i,0]**2+features[i,1]**2), np.sqrt(features[i,7]**2+features[i,8]**2)]) for i in range(len(features))])
	ind=np.array(features[:,10]> features[:,3]).astype(int)
	with warnings.catch_warnings():
		warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
		feat1 = np.array([m_jj*1e-3, features[:, 3]*1e-3, (features[:,10]-features[:, 3])*1e-3, features[:, 5]/features[:,4], features[:, 12]/features[:,11], features[:, 6]/features[:,5], features[:, 13]/features[:,12] ,features[:,-1]])
		feat2 = np.array([m_jj*1e-3, features[:, 10]*1e-3, (features[:,3]-features[:, 10])*1e-3, features[:, 12]/features[:,11], features[:, 5]/features[:,4], features[:, 13]/features[:,12], features[:, 6]/features[:,5] ,features[:,-1]])
	feat = feat1*ind+feat2*(np.ones(len(ind))-ind)
	feat = np.nan_to_num(feat)
	return feat.T

def k_fold_data_prep(args, k, direc_run, samples=None):
    print()
    if not os.path.exists(direc_run):
        os.makedirs(direc_run)

    data = file_loading(args.data_file)
    extra_bkg = file_loading(args.extrabkg_file, labels=False)

    sig = data[data[:,-1]==1]
    bkg = data[data[:,-1]==0]

    if args.signal_numner is not None:
        n_sig = args.signal_number
    elif args.signal_percentage is None:
        n_sig = 1000
    else:
        n_sig = int(args.signal_percentage*1000/0.6361658645922605)
    print("n_sig=", n_sig)

    data_all = np.concatenate((bkg,sig[:n_sig]),axis=0)
    np.random.seed(args.seed)
    np.random.shuffle(data_all)
    extra_sig = sig[n_sig:]
    innersig_mask = (extra_sig[:,0]>args.minmass) & (extra_sig[:,0]<args.maxmass)
    inner_extra_sig = extra_sig[innersig_mask]

    innermask = (data_all[:,0]>args.minmass) & (data_all[:,0]<args.maxmass)
    outermask = ~innermask
    innerdata = data_all[innermask]
    outerdata = data_all[outermask]

    extrabkg1 = extra_bkg[:312858]
    extrabkg2 = extra_bkg[312858:]

    indices = np.roll(np.array(range(5)),k)

    if args.mode=="cathode":
        print("cathode")
        if samples is None:
            samples = np.load(args.samples_file)
        if args.cathode_train_on_outer:
            samples = samples[np.logical_and(samples[:,0] > args.minmass-args.ssb_width, args.maxmass+args.ssb_width > samples[:,0])]
        if args.N_train is None:
            args.N_train = 320000
        samples_train = samples[:args.N_train+args.N_val]
        samples_test = samples[args.N_train+args.N_val:]
        print("N_train: ", len(samples_train), "; N_test: ", len(samples_test))
    elif args.mode=="cwola":
        outer_data_ssb = outerdata[np.logical_and(outerdata[:,0] > args.minmass-args.ssb_width, args.maxmass+args.ssb_width > outerdata[:,0])]
        samples_t = np.array_split(outer_data_ssb,5)
        samples_train = np.concatenate((samples_t[indices[0]], samples_t[indices[1]],samples_t[indices[2]], samples_t[indices[3]]))
        samples_test = samples_t[indices[4]]
        print("N_train: ", len(samples_train), "; N_test: ", len(samples_test))
    else:
        raise ValueError("Invalid Mode for k fold")

    print(sum(innerdata[:60000,-1]))
    print(sum(innerdata[60000:120000,-1]))
    print(sum(outerdata[:500000,-1]))
    print(sum(outerdata[500000:,-1]))

    if args.cathode_train_on_outer:
        innerdata = outerdata[np.logical_and(outerdata[:,0] > args.minmass-args.ssb_width, args.maxmass+args.ssb_width > outerdata[:,0])]		

    if args.mode in ["cwola","cathode"]:
        X_t = np.array_split(innerdata,5)
        X_train = np.concatenate((X_t[indices[0]], X_t[indices[1]], X_t[indices[2]], X_t[indices[3]]))
        X_test = X_t[indices[4]]

        X_train = np.concatenate((X_train[:,1:args.inputs+1],samples_train[:,1:args.inputs+1]),axis=0)
        Y_train = np.concatenate((np.ones(len(X_train)-len(samples_train)),np.zeros(len(samples_train))),axis=0)		

        Y_test = to_categorical(X_test[:,-1],2)
        X_test = X_test[:, 1:args.inputs+1]
        samples_test = samples_test[:,1:args.inputs+1]
    else: 
        raise ValueError('Wrong --args.mode given')

    X_train, Y_train = shuffle_XY(X_train, to_categorical(Y_train,2))

    if args.cl_norm:
        normalisation = no_logit_norm(X_train)
        X_train, _ = normalisation.forward(X_train)
        X_test, _ = normalisation.forward(X_test)
        samples_test, _ = normalisation.forward(samples_test)
    print("Train set: ", len(X_train), "; Test set: ", len(X_test))

    return X_train, Y_train, X_test, Y_test, normalisation, samples_test
