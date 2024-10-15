# Weakly Supervised Anomaly Detection: Resonant searches as cut and count experiments

This repository contains code for the following paper:

*"Weakly Supervised Anomaly Detection: Resonant searches as cut and count experiments"*, 
By Ranit Das, Thorben Finke, Marie Hein, Gregor Kasieczka, Michael Krämer, Alexander Mück and David Shih.

## Reproduction of paper results 

For the plots contained in the paper, it is necessary to loop over window numbers as well as iterate over folds as a scan across the mass spectrum with different signal region windows and k-fold cross validation ($k=5$) are used. With a bash script using slurm scripts, this can be achieved as follows: 

```
for ((window=1; window<10; window++)); do
    for ((fold=0; fold<5; fold++)); do
        sbatch pipeline_runs.slurm ${window} ${fold} 
    done
done
```

Within the slurm script a command such as the following would then be necessary: 

```
python run_pipeline.py --directory ${directory} --mode "specify_mode" --window_number ${1} --fold_number ${} --input_set ${input_set}
```

The following options are necessary for the different runs contained in the paper: 
- Baseline or $\Delta R$ dataset: ```--input_set "baseline"``` or ```--input_set "baseline" --include_DeltaR"```
- IAD or CWoLa or CATHODE: ```--mode "IAD_joep"``` or ```--mode "cwola"```or ```--mode "cathode"```
- For CATHODE, a samples file needs to be specified using ```--samples_file "file_location.npy" 
    - if different samples file are to be used per run (as is done in the paper), use: ```--samples_file_start "${samples_file_start}" --samples_file_end ${samples_file_end} --samples_file_array``` (look into code to see how looping over files is handled and modify if necessary)
- with or without signal: no additional argument required or ```--signal_number 0```
- MC estimates: ```--Herwig```
- Data-driven estimates for CATHODE: ```--cathode_on_SBs```

## Other datasets

The datasets used in this paper are 