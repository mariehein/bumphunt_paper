import numpy as np
from tqdm import tqdm

BH_percentiles = [1e-2, 1e-3, 1e-4]

def calc_and_apply_threshold(samples_preds, data_preds, efficiency):
    """
    Returns number of samples and data events before and after cut

    Apply quantile cut based on efficiency to samples classifier scores and then the
    same threshold to data classifier scores 
    """

    eps = np.nanquantile(samples_preds, 1-efficiency, method="nearest")
    if efficiency == 1:
        eps=0.
    samples_preds = np.where(samples_preds==np.nan, 0, samples_preds)
    data_preds = np.where(data_preds==np.nan, 0, data_preds)
    N_samples_after = np.size(np.where(samples_preds>=eps))
    N_samples = len(samples_preds)
    N_after = np.size(np.where(data_preds>=eps))
    N = len(data_preds)
    return N_samples_after, N_samples, N_after, N

def bump_hunt_single_window(folder, window, runs=10):
    """
    Loop over folds, apply cuts and combine results for different 
    """
    results = np.zeros((len(BH_percentiles),runs))
    true_results =  np.zeros((len(BH_percentiles),runs))
    rel_results =  np.zeros((len(BH_percentiles),runs))
    rel_error =  np.zeros((len(BH_percentiles),runs))
    stat_error = np.zeros((len(BH_percentiles),runs))

    arr_shape = (5,len(BH_percentiles), runs)
    N_samples_after = np.zeros(arr_shape)
    N_samples = np.zeros(arr_shape)
    N_after = np.zeros(arr_shape)
    N = np.zeros(arr_shape)

    for fold in range(5): 
        f = folder +"fold"+str(fold)+"/"
        samples_preds = np.load(f+"samples_preds.npy")
        samples_preds = samples_preds[-runs:]
        data_preds = np.load(f+"test_preds.npy")[-runs:]
        data_preds = np.where(data_preds==1, np.nan, data_preds)
        samples_preds = np.where(samples_preds==1, np.nan, samples_preds)
        for j, perc in enumerate(BH_percentiles):
            for i in range(len(samples_preds)):
                N_samples_after[fold, j, i], N_samples[fold, j, i], N_after[fold, j, i], N[fold, j, i] = calc_and_apply_threshold(samples_preds[i], data_preds[i], perc)
    N_samples_after = np.sum(N_samples_after, axis=0)
    N_samples = np.sum(N_samples, axis=0)
    N_after = np.sum(N_after, axis=0)
    N = np.sum(N, axis=0)
    eff_eff = N_samples_after/N_samples

    return {"epsilon_B": np.array(BH_percentiles),"N_samples_after": N_samples_after, "N_samples": N_samples,"N_after": N_after, "N": N, "epsilon_eff": eff_eff}

def bump_hunt(folder, runs=10):
    """
    Loop over windows with bump_hunt_single_window and save results
    """
    print(folder)
    exp =  np.zeros((len(BH_percentiles),9,runs))
    res = {}

    for window in range(9):
        res = bump_hunt_single_window(folder+"window"+str(window+1)+"_", window, runs=runs)
        np.savez(folder+'window'+str(window+1)+'.npz', **res)

def deltasys_single_window(folder, window, BH_percentiles, err=0, runs=10, turn_around=False):
    """
    Calculate delta_sys,n, sigma_stat and effective epsilon_B looping over folds
    """

    rel_error =  np.zeros((len(BH_percentiles),runs))
    arr_shape = (5,len(BH_percentiles), runs)
    N_samples_after = np.zeros(arr_shape)
    N_samples = np.zeros(arr_shape)
    N = np.zeros(arr_shape)
    N_after = np.zeros(arr_shape)

    for fold in range(5): 
        f = folder +"fold"+str(fold)+"/"
        samples_preds = np.load(f+"samples_preds.npy")[-runs:]
        data_preds = np.load(f+"test_preds.npy")[-runs:]
        for j, perc in enumerate(BH_percentiles):
            for i in range(len(samples_preds)):
                N_samples_after[fold, j, i], N_samples[fold, j, i], N_after[fold, j,i], N[fold, j, i] = calc_and_apply_threshold(samples_preds[i], data_preds[i], perc)
    N_samples_after = np.sum(N_samples_after, axis=0)
    N_samples = np.sum(N_samples, axis=0)
    N = np.sum(N, axis=0)
    N_after = np.sum(N_after, axis=0)
    eff_eff = N_samples_after/N_samples
    N_b_exp = eff_eff*N*(1+err)
    stat_err = np.sqrt(1/N_b_exp+1/N_samples_after)

    return (N_after-N*eff_eff)/(eff_eff*N), stat_err, eff_eff

def deltasys(folder, BH_percentiles, runs=10):
    """
    Loop over windows for deltasys_single_window and save deltasys and sigma_stat
    """

    print(folder)
    rel_error =  np.zeros((len(BH_percentiles),9,runs))
    stat =  np.zeros((len(BH_percentiles),9,runs))
    eff_eff =  np.zeros((len(BH_percentiles),9,runs))

    for window in tqdm(range(9)):
        rel_error[:,window], stat[:,window], eff_eff[:,window] = deltasys_single_window(folder+"window"+str(window+1)+"_", window, BH_percentiles, runs=runs)

    np.save(folder+"Rsys_epsB.npy", rel_error)
    np.save(folder+"stat.npy", stat)