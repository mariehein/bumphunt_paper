import numpy as np
import matplotlib.pyplot as plt

BH_percentiles = [1e-2, 1e-3, 1e-4]
plotting_direc = "plots/final/"
plotting_direc_svg = "plots/svg/"

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

def sig(N, b, err):
    """
    Complicated significance formula calculation where N=N_obs, b = N_exp and err=delta_sys
    """
    if err[0]==0:
        s=N-b
        x=N*np.log(1+s/b)-s
        x[x<0]=0
        return np.sqrt(2*(x))
    s = N - b
    ln1 = N * (b+err**2) / (b**2+N*err**2)
    ln1 = 2 * N * np.log(ln1)

    ln2 = 1 + err**2 * s / b / (b+err**2)
    ln2 = 2 * b**2 / err**2 * np.log(ln2)
    x = ln1 - ln2
    x[x<0]=0
    return np.sqrt(x)

def significances(N_after, N, N_samples_after, eff, err, err_err):
    """
    Calculate S, delta_sys,n and sigma_stat

    N_after = N_obs, N = N_SR, eff = epsilon_B, err=delta_sys, err_err=sigma_sys
    """
    N_b_exp = eff*N*(1+err)
    stat_err = np.sqrt(1/N_b_exp+1/N_samples_after)
    samples_err = np.sqrt(1/N_samples_after)
    formular_err = N_b_exp * np.sqrt(samples_err**2+err_err**2)
    rel_results = sig(N_after, N_b_exp, formular_err)
    rel_error = (N_after-N*eff)/(eff*N)
    return rel_results, rel_error, stat_err

def bump_hunt(folder, err=None, err_err=None, runs=10):
    """
    Calculate significance and delta_sys,n and sigma_stat for all 9 windows
    """

    results = np.zeros((len(BH_percentiles),9,runs))
    true_results =  np.zeros((len(BH_percentiles),9,runs))
    rel_results =  np.zeros((len(BH_percentiles),9,runs))
    rel_error =  np.zeros((len(BH_percentiles),9,runs))
    exp =  np.zeros((len(BH_percentiles),9,runs))

    if err is None:
        err = np.zeros(3)

    if err_err is None:
        err_err = err

    for window in range(9):
        res = np.load(folder+"window"+str(window+1)+".npz")
        N_after = res['N_after']
        N = res['N']
        N_samples_after = res['N_samples_after']
        eff_eff = res['epsilon_eff']
        for j, perc in enumerate(BH_percentiles):
            rel_results[j,window], rel_error[j,window], exp[j,window] = significances(N_after[j], N[j], N_samples_after[j], eff_eff[j], err[j], err_err[j])

    return rel_results, rel_error, exp

colors_results = ["blue", "red", "orange"]

def plotting(rel_results, name, min=None, max=None, plotting_directory=None, save=True):
    """
    Plot significances S    
    """

    if plotting_directory is None:
        plotting_directory=plotting_direc
    plt.figure()
    x = range(1,10)
    plt.axhline(5, color="black", linestyle="--", label="5$\sigma$")
    plt.axhline(0, color="black", label="0$\sigma$")
    for j, perc in enumerate(BH_percentiles):
        plt.errorbar(x, np.mean(rel_results[j],axis=-1), yerr = np.std(rel_results[j], axis=-1,ddof=1), label=r"$\epsilon_B$="+str(perc), fmt='o', color=colors_results[j])

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

    if save:
        plt.savefig(plotting_directory+name+".pdf")
        plt.savefig(plotting_direc_svg+name+".svg")


def plotting_error(rel_results, name, min=None, max=None, plotting_directory=None, save=True):
    """
    Plot delta_sys,n
    """

    if plotting_directory is None:
        plotting_directory=plotting_direc
    plt.figure()
    x = range(1,10)
    plt.axhline(0, color="black")
    for j, perc in enumerate(BH_percentiles):
        plt.errorbar(x, np.mean(rel_results[j],axis=-1), yerr = np.std(rel_results[j], axis=-1,ddof=1), label=r"$\epsilon_B$="+str(perc), fmt='o', color=colors_results[j])
        plt.plot(x, np.max(rel_results[j], axis=-1), 'x', color=colors_results[j])
        
    plt.grid()
    plt.ylabel(r"$\delta_{sys,n}$ (Relative systematic)")
    plt.xlabel(r"Sliding window #")
    plt.legend(loc="upper right")
    plt.subplots_adjust(bottom=0.15, left= 0.19, top = 0.92, right = 0.965)
    if min is not None and max is not None:
        plt.ylim(min,max)
    elif min is not None:
        plt.ylim(bottom=min)
    elif max is not None:
        plt.ylim(top=max)

    if save:
        plt.savefig(plotting_directory+name+"_error.pdf")
        plt.savefig(plotting_direc_svg+name+"_error.svg")