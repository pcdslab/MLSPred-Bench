# MLSPred-Bench
MLSPred-Bench: Reference EEG Benchmark for Prediction of Epileptic Seizures

MLSPred-Bench will create 12 different benchmarks based on different values of the seizure prediction horizon ($SPH$) and the seizure occurr(ence) period ($SOP$). The benchmarks are for patient-independent epileptic seizure prediction using only raw electroencephalography (EEG) data and are machine learning (ML)-ready. I.e., for each benchmark, the training, validation and test sets are each stored in a single `.hdf5` file and the corresponding labels are stored in a `.csv` file. Therefore, there are six files associated with each benchmark for a total of seventy-two files in the final ML-ready directory. For details related to the generated data, see the example usage section.

For each benchmark, MLSPred-Bench draws preictal segments of length from the $SPH$ duration. We assume there is a gap equal to the $SOP$ in minutes before the start of a seizure where the $SPH$ ends. The datasets are class-balanced where an equal amount of interictal samples are drawn from sessions of the same subject where there were no seizures. For more details, please request a pre-print version of our submitted manuscript. To execute the code, please ensure that your machine  follow the subsequent steps.

# System Requirements
System with Ubuntu 16.04 or later with at least 4 CPU cores, 64GBs of memory and greater than 150GB of available storage is recommended. Other Python 3.8.0 or higher and its built-in libraries, the code only requires the NumPy, MNE and H5PY libraries. All needed packages ae detailed in the `requirments.txt` file and can be downloaded following the setup instructions below.

# Setup
1. Check the pip version using the following command:<br>
   `pip --version`<br>
   If pip **_is installed_**, you should see an output similar to this:<br>
   ```console
   umohamma@dragon:~/virtual_environments$ pip --version
   pip 20.0.2 from /usr/lib/python3/dist-packages/pip (python 3.8)
   ```
   In that case, please skip to step 3.

2. If pip is not installed, you may get an error message or blank output screen. In that case, please run: <br>
`apt-get install python-pip` <br>\(You may need to preface the command with `sudo` and be on the sudoers list if working on a shared resource.\)

3. Ensure that venv is installed by running:<br>
   `python3 -m venv --help`<br>
   If venv is **_available_**, you may see an output similar to this:<br>
   ```console
   umohamma@dragon:~/virtual_environments$ python3 -m venv --help
   usage: venv [-h] [--system-site-packages] [--symlinks | --copies] [--clear] [--
   upgrade] [--without-pip] [--prompt PROMPT] ENV_DIR [ENV_DIR ...]

   Creates virtual Python environments in one or more target directories.

   positional arguments:
   ENV_DIR               A directory to create the environment in.

   optional arguments:
   -h, --help               show this help message and exit
   --system-site-packages   Give the virtual environment access to the system site-
   packages dir.
   --symlinks            Try to use symlinks rather than copies, when symlinks are
   not the default for the platform.
   --copies              Try to use copies rather than symlinks, even when symlinks
   are the default for the platform.
   --clear               Delete the contents of the environment directory if it
   already exists, before environment creation.
   --upgrade             Upgrade the environment directory to use this version of
   Python, assuming Python has been upgraded in-place.
   --without-pip         Skips installing or upgrading pip in the virtual environment
   (pip is bootstrapped by default)
   --prompt PROMPT       Provides an alternative prompt prefix for this environment.

   Once an environment has been created, you may wish to activate it, e.g. by
   sourcing an activate script in its bin directory.
   ```
   In that case, please skip over to step 5. If venv is **_not available_**, you may see a blank output line or an error message. In that case, please continue to step 4. 
    
5. Please run the following command:<br> 
   `apt install python3.8-venv` <br>\(Sudo permissions may be required\)
   For more information on pip, venv and virtual environments, see [working with pip and virtual environments](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/).
   
6. Create a virtual environment called mlspredbench (you may use a different name) by running the following command:<br>
`python3.8 -m venv .mlspredbench`

7. Activate the virtual environment by running:<br>
   `source .mlspredbench/bin/activate`

8. Run the command `python -m pip install --upgrade pip` to update Pip.

9. To download the repository locally, you may do it manually or run the following command:<br>
   ```sh
   git clone https://github.com/pcdslab/MLSPred-Bench.git
   cd MLSPred-Bench
   ```
   For more information, please see [managing repositories](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).
   
10. Install the required packages by running:<br>
   `python3 -m pip install -r requirements.txt`

You are now ready to run the benchmarking tool MLSPred-Bench.

# Execution
Simply run the following command:<br>
`python3 mlspred_bench_v001.py PATH/TO/RAW/TUSZ/DATA PATH/TO/SAVE/GENERATED/DATA`

As you may observe, the script requires two arguments as follows:

PATH/TO/RAW/TUSZ/DATA is the path to the raw TUSZ data which needs to be individually obtained from the [TUSZ repository](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/).

PATH/TO/SAVE/GENERATED/DATA is a valid path where the generated data can be saved.

In case the two arguments are not provided, the script will revert to default paths which will result in an error.  

# Example Usage
The script was tested on our local compute cluster called Dragon wh . The data was located in the following path: `/disk/dragon-storage/homes/eeg_data/raw_eeg_data/`. We intended to store the data in the following directory: `/disk/raptor-array/SaeedLab-Data/EEG/epilepsy/git_test_v00_mlspredb_script/`. Raptor-array is our external storage drive. To generate the benchmarks, we ran:<br>
`python3 mlspred_bench_v001.py /disk/dragon-storage/homes/eeg_data/raw_eeg_data/ /disk/raptor-array/SaeedLab-Data/EEG/epilepsy/git_test_v00_mlspredb_script/`

In the terminal, the first few lines of the output appear as follows:
```console
umohamma@dragon:~/virtual_environments$ source .mlspredbench/bin/activate
(.mlspredbench) umohamma@dragon:~/virtual_environments$ python3 mlspred_bench_v001.py /disk/dragon-storage/homes/eeg_data/raw_eeg_data/ /disk/raptor-array/SaeedLab-Data/EEG/epilepsy/git_test_v00_mlspredb_script/
Importing Libraries ... Done!

No source path given; using default path!
No target path given; using default path!
Found raw TUSZ data sub-directory!
Found TUSZ edf records directory!
Found TUSZ docs sub-directory!

TUSZ path:               /disk/dragon-storage/homes/eeg_data/raw_eeg_data/
TUSZ edf records path :  /disk/dragon-storage/homes/eeg_data/raw_eeg_data/edf
TUSZ docs path:          /disk/dragon-storage/homes/eeg_data/raw_eeg_data/DOCS/

Target base path exists!
Meta-data path exists!
Raw EEG path exists!
Montage path exists!
Interim path exists!
1-fold CV path exists!

Base path:       /disk/raptor-array/SaeedLab-Data/EEG/epilepsy/git_test_v00_mlspredb_script/
Metadata path:   /disk/raptor-array/SaeedLab-Data/EEG/epilepsy/git_test_v00_mlspredb_script/meta_data/
Raw EEG path:    /disk/raptor-array/SaeedLab-Data/EEG/epilepsy/git_test_v00_mlspredb_script/raweeg/
Montage path:    /disk/raptor-array/SaeedLab-Data/EEG/epilepsy/git_test_v00_mlspredb_script/montage/
Interim path:    /disk/raptor-array/SaeedLab-Data/EEG/epilepsy/git_test_v00_mlspredb_script/interim/
1-fold path:     /disk/raptor-array/SaeedLab-Data/EEG/epilepsy/git_test_v00_mlspredb_script/fld_sng/

TUSZ to our convention mapping:
/train/ -->   trn
/dev/   -->   vld
/eval/  -->   tst

Extracting train bi_csv meta-data ...
Extracting for patient ID aaaaafwz from session ID s001_2007_03_26. The combined session ID is trn_fwz_s001_le2.

```

Here are some of the details regarding the dirctory in which the generated data was stored and its sub-directories:
```console
umohamma@dragon:/disk/raptor-array/SaeedLab-Data/EEG/epilepsy/git_test_v00_mlspredb_script$ du -h
16M     ./meta_data
52G     ./interim
52G     ./fld_sng
111G    ./montage
88G     ./raweeg
330G    .
```

