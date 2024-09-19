import numpy as np
import matplotlib.pyplot as plt

BH_percentiles = [1e-2, 1e-3, 1e-4]
plotting_direc = "plots/final/"

plt.rcParams['pgf.rcfonts'] = False
plt.rcParams['font.serif'] = []
#plt.rcParams['text.usetex'] = True
#plt.rcParams['figure.figsize'] = 3.5, 2.625
plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['errorbar.capsize'] = 2
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16 
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.title_fontsize'] = 12
plt.rcParams['legend.fontsize'] = 12
#plt.rcParams['legend.frameon'] = True

def calc_and_apply_threshold(samples_preds, data_preds, labels, efficiency):
    eps = np.quantile(samples_preds, 1-efficiency, method="nearest")
    if efficiency == 1:
        eps=0.
    N_samples_after = np.size(np.where(samples_preds>=eps))
    N_samples = len(samples_preds)
    N_after = np.size(np.where(data_preds>=eps))#-1/3
    N = len(data_preds)
    N_bkg = np.size(np.where(data_preds[labels==0]>eps))
    N_sig = np.size(np.where(data_preds[labels==1]>eps))
    return N_samples_after, N_samples, N_after, N, N_bkg, N_sig

def sig(N, b, err):
    if err[0]==0:
        s=N-b
        x=N*np.log(1+s/b)-s
        x[x<0]=0
        return np.sqrt(2*(x))
    s = N - b
    ln1 = N * (b+err**2) / (b**2+N*err**2)
    #print(ln1)
    ln1 = 2 * N * np.log(ln1)

    ln2 = 1 + err**2 * s / b / (b+err**2)
    ln2 = 2 * b**2 / err**2 * np.log(ln2)
    #print(ln1, ln2)
    x = ln1 - ln2
    x[x<0]=0
    return np.sqrt(x)

def significances(N_after, N, N_sig, N_bkg, N_samples_after, eff, err, err_err):
    N_b_exp = eff*N*(1+err)
    stat_err = np.sqrt(1/N_b_exp+1/N_samples_after)
    samples_err = np.sqrt(1/N_samples_after)
    formular_err = N_b_exp * np.sqrt(samples_err**2+err_err**2)
    full_err = N_b_exp * np.sqrt(stat_err**2+err**2)
    results = sig(N_after, N*eff, np.zeros(N_b_exp.shape))#(N_after-N*eff)/stat_err
    rel_results = sig(N_after, N_b_exp, formular_err)#(N_after - N_b_exp)/full_err
    true_results = N_sig/np.sqrt(N_bkg)
    rel_error = (N_after-N*eff)/(eff*N)
    return results, rel_results, true_results, rel_error, stat_err

def bump_hunt_single_window(folder, window, err=None, err_err=None, runs=10, turn_around=False):
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
    N_bkg = np.zeros(arr_shape)
    N_sig = np.zeros(arr_shape)
    N_bkg_orig = 0

    for fold in range(5): 
        f = folder +"fold"+str(fold)+"/"
        #print(f)
        if turn_around: 
            samples_preds = np.load(f+"test_preds.npy")
            #print(samples_preds.shape)
            samples_preds = samples_preds[-10:]
            data_preds = np.load(f+"samples_preds.npy")[-10:]
            labels = np.zeros((data_preds.shape[1]))
        else:
            samples_preds = np.load(f+"samples_preds.npy")
            #print(samples_preds.shape)
            samples_preds = samples_preds[-10:]
            data_preds = np.load(f+"test_preds.npy")[-10:]
            labels = np.load(f+"Y_test.npy")
        N_bkg_orig += np.shape(data_preds[:,labels==0])[-1]
        for j, perc in enumerate(BH_percentiles):
            for i in range(len(samples_preds)):
                N_samples_after[fold, j, i], N_samples[fold, j, i], N_after[fold, j, i], N[fold, j, i], N_bkg[fold, j, i], N_sig[fold, j, i] = calc_and_apply_threshold(samples_preds[i], data_preds[i], labels, perc)
    N_samples_after = np.sum(N_samples_after, axis=0)
    N_samples = np.sum(N_samples, axis=0)
    N_after = np.sum(N_after, axis=0)
    N = np.sum(N, axis=0)
    #print(N)
    N_bkg = np.sum(N_bkg, axis=0)
    N_sig = np.sum(N_sig, axis=0)
    #print(N_sig)
    eff_eff = N_samples_after/N_samples

    if err is None:
        err = np.zeros(len(BH_percentiles))
        err_err = np.zeros(len(BH_percentiles))
		
    for j, perc in enumerate(BH_percentiles):
        results[j], rel_results[j], true_results[j], rel_error[j], stat_error[j] = significances(N_after[j], N[j], N_sig[j], N_bkg[j], N_samples_after[j], eff_eff[j], err[j], err_err[j])

    return results, rel_results, true_results, rel_error, stat_error, [N_after, N, N_sig, N_bkg, N_samples_after, eff_eff]

def bump_hunt(folder, err=None, err_err=None, runs=10, turn_around=False, reuse=False):
    print(folder)

    results = np.zeros((len(BH_percentiles),9,runs))
    true_results =  np.zeros((len(BH_percentiles),9,runs))
    rel_results =  np.zeros((len(BH_percentiles),9,runs))
    rel_error =  np.zeros((len(BH_percentiles),9,runs))
    exp =  np.zeros((len(BH_percentiles),9,runs))
    res = {}

    if err_err is None:
        err_err = err

    for window in range(9):
        results[:,window], rel_results[:,window], true_results[:,window], rel_error[:,window], exp[:,window], res["window"+str(window+1)] = bump_hunt_single_window(folder+"window"+str(window+1)+"_", window, err=err, runs=runs, turn_around=turn_around, err_err=err_err)
    #print(res, reuse)
    if reuse:
        return results, rel_results, true_results, rel_error, exp, res

    return results, rel_results, true_results, rel_error, exp

def bump_hunt_reuse(res, err=None, err_err=None, runs=10):

    results = np.zeros((len(BH_percentiles),9,runs))
    true_results =  np.zeros((len(BH_percentiles),9,runs))
    rel_results =  np.zeros((len(BH_percentiles),9,runs))
    rel_error =  np.zeros((len(BH_percentiles),9,runs))
    exp =  np.zeros((len(BH_percentiles),9,runs))

    if err_err is None:
        err_err = err
    for window in range(9):
        N_after, N, N_sig, N_bkg, N_samples_after, eff_eff = res["window"+str(window+1)]
        for j, perc in enumerate(BH_percentiles):
            results[j,window], rel_results[j,window], true_results[j,window], rel_error[j,window], exp[j,window] = significances(N_after[j], N[j], N_sig[j], N_bkg[j], N_samples_after[j], eff_eff[j], err[j], err_err[j])

    return results, rel_results, true_results, rel_error, exp

colors_results = ["blue", "red", "orange"]
#colors_true = ["dodgerblue", "orange", "lilac"]

def plotting(rel_results, true_results, name, min=None, max=None, plotting_directory=None):
    if plotting_directory is None:
        plotting_directory=plotting_direc
    plt.figure()
    x = range(1,10)
    plt.axhline(5, color="black", linestyle="--", label="5$\sigma$")
    plt.axhline(0, color="black", label="0$\sigma$")
    for j, perc in enumerate(BH_percentiles):
        plt.errorbar(x, np.mean(rel_results[j],axis=-1), yerr = np.std(rel_results[j], axis=-1,ddof=1), label=r"$\epsilon_B$="+str(perc), fmt='o', color=colors_results[j])
        #plt.errorbar(x, np.mean(true_results[j],axis=-1), yerr = np.std(rel_results[j], axis=-1,ddof=1), fmt='o', color=colors_true[j])

    if min is not None and max is not None:
        plt.ylim(min,max)
    elif min is not None:
        plt.ylim(bottom=min)
    elif max is not None:
        plt.ylim(top=max)

    plt.grid()
    plt.ylabel(r"Significance")
    plt.xlabel(r"Sliding window #")
    plt.legend(loc="upper right")
    plt.subplots_adjust(bottom=0.15, left= 0.19, top = 0.92, right = 0.965)

    plt.savefig(plotting_directory+name+".pdf")


def plotting_error(rel_results, name, min=None, max=None, plotting_directory=None):
    if plotting_directory is None:
        plotting_directory=plotting_direc
    plt.figure()
    x = range(1,10)
    plt.axhline(0, color="black")
    for j, perc in enumerate(BH_percentiles):
        plt.errorbar(x, np.mean(rel_results[j],axis=-1), yerr = np.std(rel_results[j], axis=-1,ddof=1), label=r"$\epsilon_B$="+str(perc), fmt='o', color=colors_results[j])
        plt.plot(x, np.max(rel_results[j], axis=-1), 'x', color=colors_results[j])
        #plt.errorbar(x, np.mean(true_results[j],axis=-1), yerr = np.std(rel_results[j], axis=-1,ddof=1), fmt='o', color=colors_true[j])
        
    plt.grid()
    plt.ylabel(r"R (relative systematic)")
    plt.xlabel(r"Sliding window #")
    plt.legend(loc="upper right")
    plt.subplots_adjust(bottom=0.15, left= 0.19, top = 0.92, right = 0.965)
    if min is not None and max is not None:
        plt.ylim(min,max)
    elif min is not None:
        plt.ylim(bottom=min)
    elif max is not None:
        plt.ylim(top=max)

    plt.savefig(plotting_directory+name+"_error.pdf")