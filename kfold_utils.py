import numpy as np
import pandas as pd
import warnings

class no_logit_norm:
    """
    Normalisation class (saves mean and standard deviation for easy application across datasets)
    """
    def __init__(self,array):
        self.mean = np.mean(array, axis=0)
        self.std = np.std(array, axis=0)

    def forward(self,array0):
    	return (np.copy(array0)-self.mean)/self.std, np.ones(len(array0),dtype=bool)

    def inverse(self,array0):
    	return np.copy(array0)*self.std+self.mean

def make_features_baseline(features, label_arr):
    """
    Make baseline feature set
    """
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
    """
    Make extended feature sets 1 and 2 from 2309.13111
    """
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

def make_features_extended3(pandas_file, label_arr):
    """
    Make extended feature set 3 from 2309.13111
    """
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

    ind_not = np.argwhere(features_j1[:,3]> features_j2[:,3])
    f_j1 = np.copy(features_j1)
    features_j1[ind_not] = features_j2[ind_not]
    features_j2[ind_not] = f_j1[ind_not]
    del f_j1
    s = np.copy(subjettinesses)
    subjettinesses[ind_not,0] = subjettinesses[ind_not,1] 
    subjettinesses[ind_not,1] = s[ind_not,0] 
    del s

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="divide by zero encountered in true_divide")
        warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")
        features = np.zeros((len(m_jj), 4+to_subjettiness*6))
        features[:,0] = m_jj * 1e-3
        features[:,1] = features_j1[:, 3] * 1e-3
        features[:,2] = (features_j2[:, 3] - features_j1[:, 3]) *1e-3
        features[:,3:3+subjettinesses.shape[-1]] = subjettinesses[:,0]
        features[:,3+subjettinesses.shape[-1]:3+subjettinesses.shape[-1]*2]= subjettinesses[:,1]
        features[:,-1] = label_arr

    return np.nan_to_num(features)

def file_loading(filename, args, labels=True, signal=0):
    """
    Load pandas data set file and calculate features used for classification as specified in args.input_set
    """
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
    elif args.input_set=="extended3":
        features = make_features_extended3(pandas_file, label_arr)
    del pandas_file
    return features

def DR(filename, labels=True):
    """
    Calculate DeltaR if args.include_DeltaR is true
    """
    if labels:
        features = np.array(pd.read_hdf(filename)[['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1_1', 'tau2j1_1', 'tau3j1_1', 'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2_1', 'tau2j2_1', 'tau3j2_1']],dtype=float)
    else: 
        features = np.array(pd.read_hdf(filename)[['pxj1', 'pyj1', 'pzj1', 'mj1', 'tau1j1_1', 'tau2j1_1', 'tau3j1_1', 'pxj2', 'pyj2', 'pzj2', 'mj2', 'tau1j2_1', 'tau2j2_1', 'tau3j2_1']],dtype=float)
        features = np.concatenate((features,np.zeros((len(features),1))),axis=1)
    Dphi = np.arccos((features[:,0]*features[:,7]+features[:,1]*features[:,8])/(np.sqrt(features[:,1]**2+features[:,0]**2)*np.sqrt(features[:,7]**2+features[:,8]**2)))
    eta1 = np.arcsinh(features[:,2]/np.sqrt(features[:,1]**2+ features[:,0]**2))
    eta2 = np.arcsinh(features[:,9]/np.sqrt(features[:,7]**2+ features[:,8]**2))
    DR = np.sqrt((Dphi)**2 + (eta1-eta2)**2)
    return DR

def k_fold_data_prep(args, samples=None):
    """ 
    Return fully preprocessed dataset separated into training and test set as well as sample test set, singal test set and training weights
    as X_train, Y_train, X_test, Y_test, samples_test, signal_test, weights_train

    Function loads data, feeds correct amount of signal into data set, initializes background template, does k-fold cross validation, and 
    normalization based on provided args. 
    """

    # File Loading and calculation of features, separation of background and signal (note Herwig contains no signal)
    data = file_loading(args.data_file, args)
    if args.mode=="IAD" or args.mode=="IAD_joep":
        extra_bkg = file_loading(args.extrabkg_file, args, labels=False)
    if args.signal_file is not None: 
        data_signal = file_loading(args.signal_file, args, labels=False, signal=1)

    if not args.Herwig:
        if args.signal_file is not None: 
            sig = data_signal
        else:
            sig = data[data[:,-1]==1]
        
    if args.include_DeltaR:
        data_DR = DR(args.data_file)
        data = np.concatenate((data[:,:args.inputs],np.array([data_DR]).T, data[:,args.inputs:]),axis=1)
        if args.mode=="IAD" or args.mode=="IAD_joep":
            extra_bkg_DR = DR(args.extrabkg_file)
            extra_bkg = np.concatenate((extra_bkg[:,:args.inputs],np.array([extra_bkg_DR]).T, extra_bkg[:,args.inputs:]),axis=1)
        if not args.Herwig:
            if args.signal_file is not None:
                sig_DR = DR(args.signal_file, labels=False)
                sig = np.concatenate((sig[:,:args.inputs],np.array([sig_DR]).T, sig[:,args.inputs:]),axis=1)
            else:
                sig = data[data[:,-1]==1]
    bkg = data[data[:,-1]==0]
    if not args.Herwig and not args.Joep:
        print(len(bkg), len(sig))
    else: 
        print(len(bkg))

    # Loading of samples files for cathode
    if args.mode=="cathode":
        samples_train = np.load(args.samples_file)
        if args.samples_ranit:
            samples_train=samples_train[:,:-1]
        if args.cathode_on_SSBs:
            mask = (samples_train[:,0]>args.minmass-args.ssb_width) & (samples_train[:,0]<args.maxmass+args.ssb_width)
            samples_train = samples_train[mask]
        samples_train = np.concatenate((samples_train, np.zeros((len(samples_train),1))), axis=1)

    # Specifying signal number and making dataset containing the correct number of signal events
    if args.signal_number is not None:
        n_sig = args.signal_number
    elif args.signal_percentage is None:
        n_sig = 1000
    else:
        n_sig = int(args.signal_percentage*1000/0.6361658645922605)
    print("n_sig=", n_sig)

    if args.randomize_signal is not None and not args.Herwig:
        np.random.seed(args.randomize_signal)
        np.random.shuffle(sig)

    if not args.Herwig:
        data_all = np.concatenate((bkg,sig[:n_sig]),axis=0)
        np.random.seed(args.set_seed)
        np.random.shuffle(data_all)
        extra_sig = sig[n_sig:]
        innersig_mask = (extra_sig[:,0]>args.minmass) & (extra_sig[:,0]<args.maxmass)
        inner_extra_sig = extra_sig[innersig_mask]
    else: 
        data_all = bkg
        np.random.seed(args.set_seed)
        np.random.shuffle(data_all)   

    # Separate data into SR and SB
    innermask = (data_all[:,0]>args.minmass) & (data_all[:,0]<args.maxmass)
    innerdata = data_all[innermask]
    outerdata = data_all[~innermask]
    np.save(args.directory+"innerdata.npy",innerdata)
    # For cathode data-driven estimate of delta_sys, switch training data to SB
    if args.mode == "cathode" and args.cathode_on_SBs:
        N = len(innerdata)
        if args.cathode_on_SSBs:
            mask = (outerdata[:,0]>args.minmass-args.ssb_width) & (outerdata[:,0]<args.maxmass+args.ssb_width)
            innerdata = outerdata[mask]
        else:
            innerdata = outerdata
        if args.select_SB_data:
            np.random.shuffle(innerdata)
            innerdata = innerdata[:N]

    if args.samples_weights is not None:
        samples_weights = np.load(args.samples_weights)

    #Make Background template for cwola and IAD
    if args.mode=="cwola":
        mask = (outerdata[:,0]>args.minmass-args.ssb_width) & (outerdata[:,0]<args.maxmass+args.ssb_width)
        samples_train = outerdata[mask]
        if args.samples_weights is not None: 
            samples_weights = samples_weights[mask] 
    elif args.mode=="IAD":
        samples_train = extra_bkg[40000:312858]
    elif args.mode=="IAD_scan":
        innerdata, samples_train = np.array_split(innerdata, 2)
        samples_train = samples_train[samples_train[:,-1]==0]
    elif args.mode=="IAD_joep":
        innermask = (extra_bkg[:,0]>args.minmass) & (extra_bkg[:,0]<args.maxmass)
        extra_bkg = extra_bkg[innermask]
        samples_train = extra_bkg[:len(innerdata)]

    #k fold cross validation help
    indices = np.roll(np.array(range(5)),args.fold_number)

    # split background template into train and test set 
    if args.mode=="cathode":
        if args.fixed_oversampling is not None:
            args.N_train = len(innerdata)*4
        print("cathode")
        samples_test = samples_train[args.N_train:]
        samples_train = samples_train[:args.N_train]
        if args.samples_weights is not None: 
            samples_weights = samples_weights[:args.N_train] 
        else: 
            samples_weights = np.ones(len(samples_train))
        print("N_train: ", len(samples_train), "; N_test: ", len(samples_test))
    else:
        if args.samples_weights is None:
            samples_weights = np.ones(len(samples_train))
        samples_t = np.array_split(samples_train,5)
        samples_w = np.array_split(samples_weights, 5)
        samples_train = np.concatenate((samples_t[indices[0]], samples_t[indices[1]],samples_t[indices[2]], samples_t[indices[3]]))
        samples_weights = np.concatenate((samples_w[indices[0]], samples_w[indices[1]],samples_w[indices[2]], samples_w[indices[3]]))
        samples_test = samples_t[indices[4]]
        print("N_train: ", len(samples_train), "; N_test: ", len(samples_test))

    X_t = np.array_split(innerdata,5)
    X_train = np.concatenate((X_t[indices[0]], X_t[indices[1]], X_t[indices[2]], X_t[indices[3]]))
    X_test = X_t[indices[4]]

    # calculate training weights
    weights_train = np.concatenate((np.ones(len(X_train)), samples_weights), axis=0)
    print("SR train:", sum(X_train[:,-1]))
    print("BT train:", sum(samples_train[:,-1]))
    X_train = np.concatenate((X_train[:,1:args.inputs+1],samples_train[:,1:args.inputs+1]),axis=0)
    Y_train = np.concatenate((np.ones(len(X_train)-len(samples_train)),np.zeros(len(samples_train))),axis=0)	

    Y_test = X_test[:,-1]
    print("SR test:", sum(Y_test))
    X_test = X_test[:, 1:args.inputs+1]
    print("BT test:", sum(samples_test[:,-1]))
    samples_test = samples_test[:,1:args.inputs+1]
    if not args.Herwig:
        signal_test = inner_extra_sig[:,1:args.inputs+1]

    inds = np.arange(len(X_train))
    np.random.shuffle(inds)
    X_train = X_train[inds]
    Y_train = Y_train[inds]
    weights_train = weights_train[inds]
    
    # normalisation of train and test set
    if args.cl_norm:
        normalisation = no_logit_norm(X_train)
        X_train, _ = normalisation.forward(X_train)
        X_test, _ = normalisation.forward(X_test)
        samples_test, _ = normalisation.forward(samples_test)
        if not args.Herwig:
            signal_test, _ = normalisation.forward(signal_test)
    print("Train set: ", len(X_train), "; Test set: ", len(X_test))

    np.save(args.directory+"Y_test.npy", Y_test)

    if args.Herwig:
        signal_test = np.ones((1,args.inputs))
    return X_train, Y_train, X_test, Y_test, samples_test, signal_test, weights_train
