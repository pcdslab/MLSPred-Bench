# MLSPred-Bench
MLSPred-Bench: Reference EEG Benchmark for Prediction of Epileptic Seizures 

# System Requirements
System with Ubuntu 16.04 or later with at least 4 CPU cores and 64GBs of memory is recommended.

# Setup
1. Check the pip version using the command. If pip is not installed, run:
   sudo apt-get install python-pip
2. Ensure that venv is installed for python 3.8. You may need to run: 
   apt install python3.8-venv (Sudo permissions may be required)
   For more informtion on pip, venv and virtual environments, see:
   https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/
3. Create a virtual environment called mlspredbench (you may use a different name) by running the following command:
4. python3 -m venv .mlspredbench
5. Activate the virtual environment by running:
   source .mlspredbecnh/bin/activate
6. To download the repository locally, you may do it manually or run the following command:
   git clone https://github.com/pcdslab/MLSPred-Bench
   For more information, please see:
   https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository
7. Install the required packages by running:
   python3 -m pip install -r requirements.txt

You are now ready to run the benchmarking tool MLSPredBench.

# Execution
Simply run the following command:
python3 mlspred_bench_v001.py PATH/TO/RAW/TUSZ/DATA PATH/TO/SAVE/GENERATED/DATA

PATH/TO/RAW/TUSZ/DATA is the path to the raw TUSZ data which needs to be individually obtained from here:
https://isip.piconepress.com/projects/nedc/html/tuh_eeg/

PATH/TO/SAVE/GENERATED/DATA is a valid path where the generated data can be saved.
