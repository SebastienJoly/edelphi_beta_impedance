# Readme

## Single simulation
The folder ```000_single_simulation``` contains an example with a single simulation. 
It can be launched by running:
 ```bash
 python 000_edelphi_impedance.py
 ```
## Parametric scan
The folder ```001_instensity_scan``` contains an example of parametric scan for htcondor.

To configure the scan you can use:
 ```bash
 python 000_config_scan.py
 ```
 
 From lxplus you can launch the simulation set by runnning:
 ```bash
 cd simulation
 bash run_htcondor
 ```
 
 An example script to plot the results can be run by:
 ```bash
 python 001_analysis_scan.py
 ```
 
 The scripts ```000_config_scan.py``` and ```001_analysis_scan.py``` can be modified to scan different parameters.
 

