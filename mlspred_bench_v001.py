
#******************************************************************************#
#-------------------------- Sec. 1: Import Libraries --------------------------#
print("Importing Libraries ...", end = " ")
#import pandas as pd 
import numpy as np
import random
import time
import h5py
import mne
import csv
import sys
import os
# from scipy.signal import butter, lfilter, freqz
# from matplotlib.patches import Rectangle
# from datetime import datetime, timedelta
# from matplotlib import pyplot as plt
# from sklearn.utils import shuffle
print("Done!\n")

#******************************************************************************#
#----------------------- Sec. 2: Define Needed Functions ----------------------#
def insertion_sort(arr, arr_name):
    for i in range(1, len(arr)):
        key = arr[i]
        key_name = arr_name[i]
        j = i-1
        while j >= 0 and key < arr[j] :
                arr[j + 1] = arr[j]
                arr_name[j + 1] = arr_name[j]
                j -= 1
        arr[j + 1] = key
        arr_name[j + 1] = key_name
    return arr, arr_name

#******************************************************************************#
#------------------------- Sec. 3: Create Target Paths ------------------------#
#raw_data_path = '/disk/dragon-storage/homes/eeg_data/raw_eeg_data/edf'   # Store raw data path in separate variable
#docs_path = "/disk/dragon-storage/homes/eeg_data/raw_eeg_data/DOCS/"     # Store path to temple docs separately

#tusz_path = "/disk/dragon-storage/homes/eeg_data/raw_eeg_data/"
#base_path = "/disk/raptor-array/SaeedLab-Data/EEG/epilepsy/git_test_tuh_eeg_v00_single_notebook/"

try:
    tusz_path = sys.argv[1]
except:
    tusz_path = "/disk/dragon-storage/homes/eeg_data/raw_eeg_data/"
    print("No source path given; using default path!")
    
try:
    base_path = sys.argv[2]
except:
    base_path = "/disk/raptor-array/SaeedLab-Data/EEG/epilepsy/git_test_v00_mlspredb_script/"
    print("No target path given; using default path!")

raw_data_path = tusz_path + 'edf'
docs_path = tusz_path + 'DOCS/'

try:
    tusz_path_list = os.listdir(tusz_path)
    print("Found raw TUSZ data sub-directory!")
except:
    print("Could not find raw TUSZ data sub-directory!")

try:
    rawd_path_list = os.listdir(raw_data_path)
    print("Found TUSZ edf records directory!")
except:
    print("Could not find TUSZ edf records directory!")
    
try:
    docs_path_list = os.listdir(docs_path)
    print("Found TUSZ docs sub-directory!")
except:
    print("Could not find TUSZ docs sub-directory!")
    
print()
print("TUSZ path:\t\t", tusz_path)
print("TUSZ edf records path :\t", raw_data_path)
print("TUSZ docs path:\t\t", docs_path)

print()
if not os.path.isdir(base_path):
    print("Creating target ML-ready storage directory ...", end =" ")
    try:
        os.mkdir(base_path)
        print("Done! Diectory Created.")
    except:
        print("Could not create directory!")
else:
    print("Target base path exists!")
    
meta_path = base_path + "meta_data/"
if not os.path.isdir(meta_path):
    print("Creating meta-data storage sub-directory ...", end =" ")
    os.mkdir(meta_path)
    print("Done! Diectory Created.")
else:
    print("Meta-data path exists!")
    
data_path = base_path + "raweeg/"
if not os.path.isdir(data_path):
    print("Creating raw EEG storage sub-directory ...", end =" ")
    os.mkdir(data_path)
    print("Done! Diectory Created.")
else:
    print("Raw EEG path exists!")
    
mont_path = base_path + "montage/"
if not os.path.isdir(mont_path):
    print("Creating montage sub-directory ...", end =" ")
    os.mkdir(mont_path)
    print("Done! Diectory Created.")
else:
    print("Montage path exists!")
    
intr_path = base_path + "interim/"
if not os.path.isdir(intr_path):
    print("Creating interim directory ...", end =" ")
    os.mkdir(intr_path)
    print("Done! Directory Created.")
else:
    print("Interim path exists!")
    
sf00_path = base_path + "fld_sng/"
if not os.path.isdir(sf00_path):
    print("Creating single-fold directory ...", end =" ")
    os.mkdir(sf00_path)
    print("Done! Directory Created.")
else:
    print("1-fold CV path exists!")

print()
print("Base path:\t", base_path)
print("Metadata path:\t", meta_path)
print("Raw EEG path:\t", data_path)
print("Montage path:\t", mont_path)
print("Interim path:\t", intr_path)
print("1-fold path:\t", sf00_path)

#******************************************************************************#
#----- Sec. 4: Identify and print statistics related to available records -----#

dir_name_list = ['/train/',  '/dev/', '/eval/']
id_type_prefix = {} # Create empty prefix dictionary
# populate according to type of data tested; remove slashes from key
print("\nTUSZ to our convention mapping:")
for name, prefix in zip(dir_name_list,['trn', 'vld', 'tst']):
    print(name + '\t-->   ' + prefix) # print name to prefix conversion
    id_type_prefix[name[1:-1]] = prefix # Store prefix value in dict.; key has no slashes
print()

all_ids_list = []      # List of all ID's
all_rtp_list = []      # Not used
all_edf_fls_list = []  # List of complete paths to all edf files
all_csv_fls_list = []  # List of complete paths to all csv files
all_ids_dict = {}      # Dictionary of reference types per dataset type ("trn", "vld" or "tst")
all_rtp_dict = {}      # Not used
all_edf_fls_dict = {}  # Dictionary of edf file names per dataset type ("trn", "vld" or "tst")
all_csv_fls_dict = {}  # Dictionary of csv file names per dataset type ("trn", "vld" or "tst")
edf_file_pth_dict = {} # Dictionary of full edf file paths per dataset type ("trn", "vld" or "tst")
csv_file_pth_dict = {} # Dictionary of full csv file paths per dataset type ("trn", "vld" or "tst")
list_rec_paths_more_than_one = [] # A list of sessions with more than 1 record (there are none)
label_paths_count_dict = {'ar1': 0, 'le2': 0, 'ar3': 0, 'le4': 0} # Dict. to count number of paths for each ref. type
label_files_count_dict = {'ar1': 0, 'le2': 0, 'ar3': 0, 'le4': 0} # Dict. to count number of files for each ref. type

for path_type in dir_name_list: # Loop 1
    path_key = path_type[1:-1]  # Store the dtype directory name (['/train/',  '/dev/', '/eval/']) without slashes
    path_name = raw_data_path + path_type     # The full path is base path plus the complete path of dataset type
    path_list = sorted(os.listdir(path_name)) # List all the directories (at the patient level)
    all_ids_dict[path_key] = []               # Build up the dictionary of ID's
    all_edf_fls_dict[path_key] = []           # List of all edf files in the patient path
    all_csv_fls_dict[path_key] = []           # List of all csv files in the patient path
    print(f'Extracting {path_key} bi_csv meta-data ...')
    for p in path_list:                       # Loop 2
        patient_id = p                        # get the complete patient ID including first 5 "a's"
        sessions_path = path_name+p           # create a session path
        session_list = sorted(os.listdir(sessions_path)) # a list of session paths for each patient
        sessions_path_list = []               # Build the session path
        for s in session_list: 
            sessions_path_list.append(sessions_path+'/'+s) 
        for s in sessions_path_list:          # Loop 3
            session_id = s.split('/')[-1] 
            recordings_path = sorted(os.listdir(s))
            if len(recordings_path)>1:
                list_rec_paths_more_than_one.append(s)
            for r in recordings_path:         # Loop 4
            #r = recordings_path[-1]
                temp_rec_id = r.split('_')
                rec_suffix = f'{temp_rec_id[-(len(temp_rec_id)//2)]}{temp_rec_id[0][-1]}'
                label_paths_count_dict[rec_suffix] += 1
                all_ids_list.append(f'{id_type_prefix[path_key]}_{patient_id[-3:]}_{session_id[:4]}_{rec_suffix}')
                all_ids_dict[path_key].append(all_ids_list[-1])
                print(f'Extracting for patient ID {patient_id} from session ID {session_id}.', end = " ")  
                print(f'The combined session ID is {all_ids_list[-1]}.', end  = "\r") 
                edf_file_pth_dict[all_ids_list[-1]] = []
                csv_file_pth_dict[all_ids_list[-1]] = []
                temp_edf_list = []
                temp_csv_list = []
                files_list = sorted(os.listdir(s+'/'+r))
                for f in files_list:          # Loop 5
                    file_path = s + '/'+ r + '/' + f
                    if file_path.endswith(".edf"):
                        all_edf_fls_list.append(file_path)
                        all_edf_fls_dict[path_key].append(file_path)
                        temp_edf_list.append(file_path)
                    if file_path.endswith(".csv_bi"):
                        all_csv_fls_list.append(file_path)
                        all_csv_fls_dict[path_key].append(file_path)
                        temp_csv_list.append(file_path)
                edf_file_pth_dict[all_ids_list[-1]] = temp_edf_list
                csv_file_pth_dict[all_ids_list[-1]] = temp_csv_list
                label_files_count_dict[rec_suffix] += len(temp_edf_list)
        time.sleep(0.1)
    print(f'\nDone for {path_key} data.')
    print()
    
print(f'There are {len(edf_file_pth_dict.keys())} sub-directories comprising EDF files.')
print(f'There are {len(csv_file_pth_dict.keys())} sub-directories comprising CSV files.')
print(f'There are {len(all_ids_list)} ID\'s of the form typePrefix_patientID_SessionID_montageType.')
print()
print(f'There are {len(all_edf_fls_list)} distinct EDF file paths in the all file path list for EEG.')
sumD = 0
for d in edf_file_pth_dict.values():
    sumD += len(d)
print(f'There are {sumD} sum elements across all lists in the EDF file path dictionary for EEG.')
print(f'There are {len(all_csv_fls_list)} distinct CSV file paths in the all file path list for EEG.')
sumC = 0
for d in csv_file_pth_dict.values():
    sumC += len(d)
print(f'There are {sumC} sum elements across all lists in the CSV file path dictionary for EEG.')

print()
print("These files have more than one reference type:")
for x in list_rec_paths_more_than_one:
    print(x)

print()
for x in label_paths_count_dict:
    print(f'Label: {x} Count: {label_paths_count_dict[x]:4d}', end ="\t\t")
print()
for x in label_files_count_dict:
    print(f'Label: {x} Count: {label_files_count_dict[x]:4d}', end ="\t\t")
ar_file_count = label_files_count_dict['ar1']+label_files_count_dict['ar3']
le_file_count = label_files_count_dict['le2']+label_files_count_dict['le4']
print(f'\nThere are {ar_file_count} AR type files and {le_file_count} LE types.\n')

#******************************************************************************#
#----- Sec. 5: Extract TUH metadata and convert it to user-friendly form ------#

meta_data_dict = {} # Define an empty meta_data_dict
for dkey in all_ids_list: # Go over each customized session ID
    csv_file_names_list = csv_file_pth_dict[dkey]  # Populate a list of all csv files of that session
    meta_data_dict[dkey] = [] # intialize an empty list for each session
    for f in csv_file_names_list:
        recording_number_str = f.split('.')[0][-3:] # Identify the recording number from the file name
        # Print progress on a single line
        print(f'Extracting meta data for patient ID {dkey:s} recording # {recording_number_str} ...', end = " ")
        meta_data_dict[dkey].append(f) # Append the csv file name to the list
        print(f"Saved {len(meta_data_dict):04d}/{len(all_ids_list)} lists.", end = '\r')
        #time.sleep(0.1)
print("\nDone extracting Metadata for all sessions of each patient in train, dev and eval directories.")

print("\nGenerating and saving metadata for patient/session:")
all_sessions_duration_list = [] # List of durations of all sessions
all_sessions_cumldurs_list = [] # List of cumulative durations of all sessions
all_sessions_duration_dict = {} # Dictionary of durations of each recording in a single session
all_sessions_cumldurs_dict = {} # Dict of cumulative durations of all sessions
for dkey in all_ids_list:
    #print(dkey, end = " ")
    csv_file_names_list = csv_file_pth_dict[dkey] # Get the csv files for each customized session ID
    if len(csv_file_names_list) > 0: # proceed if we have at least one recording in a session
        pat_id = csv_file_names_list[0].split('/')[-4] 
        ses_id = csv_file_names_list[0].split('/')[-3]
    all_sessions_duration_dict[dkey] = [] # Create an empty list ot record durations of each session
    all_sessions_cumldurs_dict[dkey] = [] # Define empty list for cumulative durations; can be initialized to zero too.
    X = f'Patient: {pat_id} Session: {ses_id}\n'
    for f in csv_file_names_list:
        with open(f) as my_csv:
            text_blob = my_csv.read().split('\n') # get 
            file_name_edf = f.split('/')[-1][:-7]+'.edf'
            file_dirs_edf = '/'.join(f.split('/')[-2:-1])+'/'+file_name_edf
            cur_dur = float(text_blob[2].split('=')[-1].strip()[:-5])
            X += f'\nFile Name: {file_dirs_edf}\n'
            X += f'File Start Time: 0.0000\n'
            X += f'File End Time: {cur_dur:0.4f}\n'
            count_seizures = 0
            for t in text_blob[6:-1]:
                if t.split(',')[3] == 'seiz':
                    count_seizures += 1
            X += f'Number of Seizures in File: {count_seizures}\n'
            for t in text_blob[6:-1]:
                cur_lab = t.split(',')[3]
                lab_strt_time = float(t.split(',')[1])
                lab_ends_time = float(t.split(',')[2])
                #print(f' {cur_lab} {lab_strt_time:7.2f} {lab_ends_time:7.2f}', end =" ")
                if t.split(',')[3] == 'seiz':
                    seiz_strt_time = float(t.split(',')[1])
                    seiz_ends_time = float(t.split(',')[2])
                    X += f'Seizure Start Time: {seiz_strt_time:.4f} seconds\n'
                    X += f'Seizure End Time: {seiz_ends_time:.4f} seconds\n'
    write_file_name = f'{meta_path}/{dkey}.txt'
    with open(write_file_name, 'w') as wrf:
        wrf.write(X)
    print(dkey, X.split('\n')[0], end = "\r")
    time.sleep(0.1) # Sleep for pretty printing
print("\nDone saving Metadata for all sessions of each patient in train, dev and eval directories.\n")

#******************************************************************************#
#--- Sec. 6: Identify the channels and sampling rates used for each session ---#

'''
In the following cell, we are going to make a create the following: 
1. A set of all sampling rates observed across 7,377 edf records
2. A list of frequency of occurrence of each sampling rate
3. A set of all channels from the required montage channels that are observed
4. A set of counts of number of channels that are present among the different records
5. A list of frequency of occurrence of each channel count
6. A dictionary of lists of sampling rates per file <br>
    A. May be a dictionary of dictionaries or a simple list
7. A dictionary of lists of channel sets per session
8. A dictionary of lists of channel counts per session

We will build up the sampling rate set as we go along, and add to the count list each 
time there is a new sampling rate and intialize the count to 1. Once the count is initialized, 
each time that particular sampling rate is observed, we append the corresponding count. 
A similar strategy is employed for the channel names and the channel counts. 

Items or features that will be added soon:
1. A dictionary of each channel observed and the number of time it appears; to populate missing channels 
2. The number of times the same session has a different sampling rate over multiple edf records
3. A dictionary of channels belonging to each session (similar to sampling rate)
    A. May be a dictionary of dictionary or a dictionary of lists
4. An analysis of patients where the channel numbers differ
'''

print("Figuring out channel names and sampling rates...")

channels_to_include = ['EEG T6-REF', 'EEG T5-REF', 'EEG T4-REF', 'EEG T3-REF', 
                       'EEG P4-REF', 'EEG P3-REF', 'EEG O2-REF', 'EEG O1-REF', 
                       'EEG FP2-REF', 'EEG FP1-REF', 'EEG F8-REF', 'EEG F7-REF', 
                       'EEG F4-REF', 'EEG F3-REF', 'EEG CZ-REF', 'EEG C4-REF', 
                       'EEG C3-REF', 'EEG T6-LE', 'EEG T5-LE', 'EEG T4-LE', 
                       'EEG T3-LE', 'EEG P4-LE', 'EEG P3-LE', 'EEG O2-LE', 
                       'EEG O1-LE', 'EEG FP2-LE', 'EEG FP1-LE', 'EEG F8-LE', 
                       'EEG F7-LE', 'EEG F4-LE', 'EEG F3-LE', 'EEG CZ-LE', 
                       'EEG C4-LE', 'EEG C3-LE']

channels_count_ar_dict = {}
channels_count_le_dict = {}
for ch in channels_to_include:
    if ch[-3:] == 'REF':
        channels_count_ar_dict[ch] = 0
    if ch[-2:] == 'LE':
        channels_count_le_dict[ch] = 0
for ch in channels_count_ar_dict:
    print(ch, end = "\t")
print("\n")
for ch in channels_count_le_dict:
    print(ch, end = "\t")
print()

time_data_dict = {}
amps_data_dict = {}
chan_name_dict = {}
final_eeg_file_list = []
final_eeg_file_dict = {}
sampling_rate_dict = {}
nmbr_of_samps_dict = {}
duration_edfs_dict = {}
totl_duration_dict = {}
totl_num_samp_dict = {}
totl_duration_list = []
totl_num_samp_list = []
sampling_rate_set = set()
chan_names_set = set()
chan_count_set = set()
sampling_rate_list = []
sampling_rate_cnts = []
channels_freq_list = []
channels_freq_cnts = []
chan_count_lst = []
count_files = 0
print_abv_pat_extra_line_flag = False
for dkey in all_ids_list:
    if print_abv_pat_extra_line_flag:
        print()
    print(dkey, end = "\r")
    print_abv_pat_extra_line_flag = False
    edf_file_names_list = edf_file_pth_dict[dkey]
    num_of_edf_recs = len(edf_file_names_list)
    time_data_list = []
    amps_data_list = []
    chan_name_list = []
    sampling_rate_dict[dkey] = []
    duration_edfs_dict[dkey] = []
    nmbr_of_samps_dict[dkey] = []
    totl_duration_dict[dkey] = []
    totl_num_samp_dict[dkey] = 0
    totl_duration_dict[dkey] = 0
    totl_num_samp_list.append(0)
    totl_duration_list.append(0)
    for f in edf_file_names_list:
        count_files += 1
        print_ch_extra_line_flag = True
        recording_number_str = f.split('.')[0][-3:]
        X1 = mne.io.read_raw_edf(f,include=channels_to_include, verbose="Warning")
        old_samprte_set = sampling_rate_set.copy()
        old_chancnt_set = chan_count_set.copy()
        sampling_rate_tmp = X1.info["sfreq"]
        sampling_rate_set.add(sampling_rate_tmp)
        ch_names_list = X1.ch_names
        chan_count_lst.append(len(ch_names_list))
        chan_count_set.add(len(ch_names_list))
        srate_set_diff = sampling_rate_set-old_samprte_set
        chans_set_diff = chan_count_set-old_chancnt_set
        nmbr_of_time_inst = X1.n_times
        dur_of_rw_edf_rec = nmbr_of_time_inst/sampling_rate_tmp
        sampling_rate_dict[dkey].append(sampling_rate_tmp)
        nmbr_of_samps_dict[dkey].append(nmbr_of_time_inst)
        duration_edfs_dict[dkey].append(dur_of_rw_edf_rec)
        totl_num_samp_dict[dkey] += nmbr_of_time_inst
        totl_duration_dict[dkey] += dur_of_rw_edf_rec
        for ch in ch_names_list:
            chan_names_set.add(ch)
            if ch[-3:] == 'REF':
                channels_count_ar_dict[ch] += 1
            if ch[-2:] == 'LE':
                channels_count_le_dict[ch] += 1
        if len(srate_set_diff) == 1:
            print()
            print(f"New Sampling Rate Observed: {sampling_rate_set}")
            sampling_rate_list.append(int(srate_set_diff.pop()))
            sampling_rate_cnts.append(0)
            print_ch_extra_line_flag = False
            print_abv_pat_extra_line_flag = True
        if len(chans_set_diff) == 1:
            if print_ch_extra_line_flag:
                print()
            print(f"New Channel Count Observed: {chan_count_set}")
            channels_freq_list.append(chans_set_diff.pop())
            channels_freq_cnts.append(0)
            print_abv_pat_extra_line_flag = True           
        for i in range(len(sampling_rate_list)):
            if int(X1.info["sfreq"]) == sampling_rate_list[i]:
                sampling_rate_cnts[i] += 1
        for i in range(len(channels_freq_list)):
            if len(X1.ch_names) == channels_freq_list[i]:
                channels_freq_cnts[i] += 1
print("Done with channel names and sampling rates!")

print("Set of sampling rates:", sampling_rate_set)
print("Set of channels names:", chan_names_set)

print("\nUnordered set of sampling rates observed:", sampling_rate_set)
print("Unordered set of number of channels observed:", chan_count_set)
print("\nSampling Rates List:", end = "\t")
sampling_rate_list_sorted, sampling_rate_cnts_sorted = insertion_sort(sampling_rate_list, sampling_rate_cnts)
channels_freq_list_sorted, channels_freq_cnts_sorted = insertion_sort(channels_freq_list, channels_freq_cnts)
for x in sampling_rate_list_sorted:
    print(f"{x:4d}", end = "\t")
print("\nSampling Rates Count:", end = "\t")
for x in sampling_rate_cnts_sorted:
    print(f"{x:4d}", end = "\t")
print()
print("\nNumber of Ch. List:", end = "\t")
for x in channels_freq_list_sorted:
    print(f"{x:4d}", end = "\t")
print("\nNumber of Ch. Count:", end = "\t")
for x in channels_freq_cnts_sorted:
    print(f"{x:4d}", end = "\t")
print()

ar_keys_list = list(channels_count_ar_dict.keys())
ar_vals_list = list(channels_count_ar_dict.values())
ar_vals_list_sort, ar_keys_list_sort = insertion_sort(ar_vals_list, ar_keys_list)
le_keys_list = list(channels_count_le_dict.keys())
le_vals_list = list(channels_count_le_dict.values())
le_vals_list_sort, le_keys_list_sort = insertion_sort(le_vals_list, le_keys_list)
channels_count_ar_dict_sorted = {}
channels_count_le_dict_sorted = {}
for x, y in zip(ar_keys_list_sort, ar_vals_list_sort):
    channels_count_ar_dict_sorted[x] = y
for x, y in zip(le_keys_list_sort, le_vals_list_sort):
    channels_count_le_dict_sorted[x] = y
for x, y in zip(channels_count_ar_dict_sorted, channels_count_le_dict_sorted):
    print(f"{{{x:>11s}: {channels_count_ar_dict_sorted[x]}}}, {{{y:>10s}: {channels_count_le_dict_sorted[y]}}}")
print()
#******************************************************************************#
#--------- Sec. 7: Extract each session's raw EEG data and concatenate --------#

strt_indexes_dict = {}
ends_indexes_dict = {}
for dkey in nmbr_of_samps_dict:
    strt_indexes_dict[dkey] = [0]
    for x in nmbr_of_samps_dict[dkey]:
        strt_indexes_dict[dkey].append(x+strt_indexes_dict[dkey][-1])

final_ids_with_data_list = []
for dkey in all_ids_list:
    edf_file_names_list = edf_file_pth_dict[dkey]
    num_of_edf_recs = len(edf_file_names_list)
    temp_array = np.zeros((17, totl_num_samp_dict[dkey]))
    for i in range(num_of_edf_recs):
        f = edf_file_names_list[i] 
        recording_number_str = f.split('.')[0][-3:]
            
        X1 = mne.io.read_raw_edf(f,include=channels_to_include, verbose="Warning")
        Y = X1._read_segment()
        try:
            assert Y.shape[1] == nmbr_of_samps_dict[dkey][i]
        except:
            print("Could not match for: ", dkey)
        strt_index = strt_indexes_dict[dkey][i]
        ends_index = strt_indexes_dict[dkey][i+1]
        try:
            temp_array[:,strt_index:ends_index] = Y
        except:
            temp_array = None
            print("Could not assign for: ", dkey)
    if not temp_array is None and temp_array.shape[1] > 0:
        final_ids_with_data_list.append(dkey)
        np.save(data_path+dkey+'.npy', temp_array)        
        print(f"Saving concatenated raw EEG Data for patient ID {dkey:s} record # {i+1:02d}/{num_of_edf_recs:02d}",
              f"... Done for {all_ids_list.index(dkey)+1}/{len(all_ids_list)}", end = '\r')  
    else:
        print(200*" ", end ="\r")
        print(f"Patient aaaaa{dkey[:3]} Session {dkey[8:12]} is EMPTY!")
        
#******************************************************************************#
#----------- Sec. 8: Calculate Montages from raw EEG data and store -----------#
'''
In this next cell, we try to do several tasks leading to the creation of montages.
We read the reference type of the raw EEG recording from the file name, and figure 
out which channel to use. 

In the following cell, we figure out the orders of the channels, and use a dictionary 
structure to map each channel to a specific corresponding row in the ".npy" file.

First, we create a mapping from channel to row number for each reference type: LE and AR. 
Next, we use the montage description to define the "difference" pairs as a list of tuples. 
In total, there are 22 lists.
'''
montage_lst_tcp_ar3 = []
channel_lst_tcp_ar3 = []
channel_set_tcp_ar3 = set()
with open(docs_path + "03_tcp_ar_a_montage.txt") as f:
    for lines in f.readlines():
        if lines[:-1][:7] == "montage":
            temp_ch1 = "EEG " + lines[:-1].split(' ')[-1]
            temp_ch2 = "EEG " + lines[:-1].split(' ')[-4]
            temp_ch1 = temp_ch1.split('-')[0]
            temp_ch2 = temp_ch2.split('-')[0]
            # print(temp_ch1, temp_ch2)
            channel_lst_tcp_ar3.append((temp_ch2, temp_ch1))
            channel_set_tcp_ar3.add(temp_ch2)
            channel_set_tcp_ar3.add(temp_ch1)  

print("AR3 montage (lowest common number of channels)")
for x in channel_lst_tcp_ar3:
    print(f'{x[0]:7s} - {x[1]:>7s}', end = "     ")
print("\n")

channels_to_include = ['EEG T6', 'EEG T5', 'EEG T4', 'EEG T3', 'EEG P4', 'EEG P3', 'EEG O2', 'EEG O1', 'EEG FP2', 
                       'EEG FP1', 'EEG F8', 'EEG F7', 'EEG F4', 'EEG F3', 'EEG CZ', 'EEG C4', 'EEG C3']

channel_index_dict = {}
count_chs = 0   
for ch in channels_to_include:
    channel_index_dict[ch] = count_chs
    count_chs += 1

print("Channel Mapping:")
for ch in channel_index_dict:
    print(f"{ch:>7}: {channel_index_dict[ch]:2d}", end = "\t")
print("\n")

montage_rows_tuple_list = []
for ch in channel_lst_tcp_ar3:
    montage_rows_tuple_list.append((channel_index_dict[ch[0]], channel_index_dict[ch[1]]))
print("Montage length:", len(montage_rows_tuple_list))

count_keys = 0
for dkey in all_ids_list:
    count_keys += 1
    file_path = data_path + dkey + '.npy'
    mont_file_path = mont_path + dkey + '.npy'
    try:
        temp_read_array = np.load(file_path)
    except FileNotFoundError:
        print(100*" ", end = "\r")
        print(f"File for session {dkey} not found!")
    try:
        assert temp_read_array.shape[0] == 17 and temp_read_array.shape[1] > 0
    except:
        print("Could not assert one or more of data or number of channels!")
    
    
    temp_write_array = np.zeros((20, temp_read_array.shape[1]))
    for i in range(len(montage_rows_tuple_list)):
        ind_ch1 = montage_rows_tuple_list[i][0]
        ind_ch2 = montage_rows_tuple_list[i][1]
        temp_write_array[i,:] = temp_read_array[ind_ch1,:]-temp_read_array[ind_ch2,:]
    try:
        np.save(mont_file_path, temp_write_array)
    except:
        print("\nCould not save!")
    
        
    print(f"For ID {dkey}, saved data of shape: {temp_read_array.shape[0]:2d} x {temp_read_array.shape[1]:<12d}",
          f"Progress {count_keys}/{len(all_ids_list)}", end = "\r")

#******************************************************************************#
#------------ Sec. 9: Analyse the metadata to build the benchmarks ------------#
'''
In this notebook, we use the previously created metadata files to extract the raw data 
from the set of edf files in each session, concatenate the raw data from each session 
into a single long matrix and then store it in NumPy ('.npy') format. 

In the following cells, we store the text blobs for each session in a dictionary where 
the keys are the complete session ID's. Because the session ID is also the name of the 
text file comprising the metadata for that session, it is easy to extract the complete 
session ID as defined prior. The complete session ID is comprised of four fields, each 
separated by an underscore. 
    
The first field comprises three letters that indicate the dataset from which it was taken 
(training - trn, testing - tst or development - vld because we consider the development 
set to be the validation dataset. The second field identifies the patient ID using only 
the last three alphabets of the anonymized patient ID. Only these 3 letters are unique 
out of the 8 letter alphabetical ID, the first 5 are alays 'aaaaa' for all patients. 
The third field identifies the session number with 4 characters. The first character is 
always 's' followed by 3 characters of an integer number in 03d format starting from 001. 
    
The last field identifies the electrode reference type using where there are four different 
possible configurations. The 2 reference types used are the average referenced (AR) or 
linked ear (LE) reference and in each reference type, some patients have 2 electrodes 
(the A1-Ref and A2-Ref) which leads to a total of 4 possibilities. AR1 and LE2 are 
respectively configurations where electrodes are available.    
    
The text blobs are stored in a dictionary called 'txt_blks_annot_dict'. The keys to the 
dictionary will simply be the session ID's. In another dictionary called 'ind_lsts_nwlns_dict', 
we also store the indexes of each newline character that occurs in the text file. This was 
used for the parser developed for CHB-MIT but is obsolete now. We keep this variable in 
case it is needed in the future. We also store a list of customized patient ID's which 
is simply the first 2 fields of each sesison ID.

Next, we parse the text blobs to extract the following useful information for each session: 
the name of each edf file for each record, the duration of that particular recording, the 
number of seizures obsereved and the start and end times of each seizure. 

Based on this information, we iteratively determine the cumulative duration of a session 
comprising multiple recordings and for each record, we store a 'cumulative' start time of 
a seizure. Let $t_{start}$ denote the start time of a seizure and $t_{end}$ be the end of 
the ictal duratuion. For example, let us consider 2 recordings, one 2400 seconds long and 
the second 3600 seconds long. If seizure 1 has a labeled start time of 1800 seconds in 
recording 1, the start time of seizure 1 is 1800 seconds. However, if seizure 2 has a 
labeled start time of 1800 seconds, the actual start time is 2400 + 1800 = 4200 seconds. 
We use the same concept to track the seizure end times. We also keep track of the total 
number of sessions encountered, the total number of recordings across all records and 
the total number of seizures observed.

Define the metadata paths and the actual data path 
(which is one directory below the metadata path).
'''

# List of metadata text files
txt_file_names_list = sorted(os.listdir(meta_path))
print("The names of the first 10 files are:")
for x in txt_file_names_list[:10]:
    print(x, end = "\t")
print()

# List of full path names pointing to each metadata file
txt_file_dirs_list = []
for file in txt_file_names_list:
    txt_file_dirs_list.append(os.path.join(meta_path, file))
print("\nThe full paths to the first 10 files:")
for x in txt_file_dirs_list[:10]:
    print(x)
    
# Extract text blobs and new line indexes
txt_blks_annot_dict = {} 
ind_lsts_nwlns_dict = {}
edf_file_numbs_dict = {}
list_of_keys = []
for fname in txt_file_dirs_list:
    key = fname.split('/')[-1][:-4]
    list_of_keys.append(key)
    with open(fname,'r') as f:
        sum_txt = f.readlines() 
        txt_blks_annot_dict[key] = sum_txt
    ind_lsts_nwlns_dict[key] =[i for i in range(len(sum_txt)) if sum_txt[i] == '\n']

# Extract the set of patient ID's
pat_ids_set = set()
for x in txt_file_names_list:
    pat_ids_set.add(x[:7])
pat_ids_lst = list(pat_ids_set)
pat_ids_lst.sort()
print("The first 8 patient ID's are:")
for y in pat_ids_lst[:8]:
    print(y, end = "\t\t")
print()

cnt_sess_with_seiz_list = [] # Count the number of sessions with seizures
cnt_sess_zero_seiz_list = [] # Count the number of sessions with NO seizures
cnt_seizs_per_sess_list = [] # Store the number of seizures per session in a list
cnt_seizs_per_sess_dict = {} # Store the number of seizures per session in a dict
cum_time_with_seiz_list = [] # Store the number of seizures per session in a list
pid_sess_with_seiz_list = [] # Store the ID's of patients with a seizure
pid_sess_zero_seiz_list = [] # Store the ID's of patients with NO seizures
seiz_cum_strt_time_list = [] # Store seizure cumulative start times in a list and dict
seiz_cum_strt_time_dict = {'trn':[], 'vld':[], 'tst':[]} # Store train, validate, test start times independently
seiz_cum_ends_time_list = [] # Store seizure cumulative end times in a list and dict
seiz_cum_ends_time_dict = {'trn':[], 'vld':[], 'tst':[]} # Store train, validate, test end times independently
seiz_cum_strt_time_dict_unsorted = {} 
seiz_cum_ends_time_dict_unsorted = {}
cnt_tot_sess = 0 # track count of total sessions
cnt_tot_recs = 0 # track count of edf records
cnt_tot_seiz = 0 # track count of all seizures
for dkey in list_of_keys:
    cnt_tot_sess += 1     #                      # Increase session count
    cnt_seiz_per_sess = 0
    cnt_recs_per_sess = 0
    cum_tot_time = 0
    seiz_cum_strt_time_list.append([])          # Initialize cumulative start time dict to an empty list
    seiz_cum_ends_time_list.append([])          # Initialize cumulative end time dict to an empty list
    seiz_cum_strt_time_dict_unsorted[dkey] = [] #
    seiz_cum_ends_time_dict_unsorted[dkey] = [] #  
    for x in txt_blks_annot_dict[dkey]:         # Iterate over all text blocks
        if x[:9] == 'File Name':                # If a line starts with "File Name" (in the metadata file)
            cnt_tot_recs += 1                   # Increment the number of records
            cnt_recs_per_sess += 1              # Increment the number of records per session
        #if x[:10] == 'File Start':
            #cnt_tot_recs += 1
        if x[:18] == 'Number of Seizures':           # If a line starts with Number of Seizures (first 8 characters)
            num_of_seiz_in_rec = int(x.split(' ')[-1].strip()) # Last field after splitting with spaces gives seizure count
            cnt_seiz_per_sess += num_of_seiz_in_rec  # Append the seizure count to the variable tracking cont per seizure
        if x[:13] == 'Seizure Start':                                           # if line starts with "Seizure Start"
            seiz_strt_time = float(x.split(' ')[-2].strip())                    # Current seizure start time
            seiz_cum_strt_time_list[-1].append(seiz_strt_time+cum_tot_time_old) # Added to previous end time and appended 
        if x[:11] == 'Seizure End':
            seiz_ends_time = float(x.split(' ')[-2].strip())
            seiz_cum_ends_time_list[-1].append(seiz_ends_time+cum_tot_time_old)
        if x[:8] == 'File End':
            end_time = float(x.split(' ')[-1].strip())
            cum_tot_time_old = cum_tot_time
            cum_tot_time += end_time
    seiz_cum_strt_time_dict[dkey[:3]].append(seiz_cum_strt_time_list[-1])
    seiz_cum_ends_time_dict[dkey[:3]].append(seiz_cum_ends_time_list[-1])
    seiz_cum_strt_time_dict_unsorted[dkey].append(seiz_cum_strt_time_list[-1])
    seiz_cum_ends_time_dict_unsorted[dkey].append(seiz_cum_ends_time_list[-1])
    if(cnt_seiz_per_sess) > 0:
        cnt_sess_with_seiz_list.append(cnt_seiz_per_sess)
        pid_sess_with_seiz_list.append(dkey)
        cum_time_with_seiz_list.append(cum_tot_time)
        cnt_seizs_per_sess_list.append(cnt_seiz_per_sess)
    else:
        pid_sess_zero_seiz_list.append(dkey)
    cnt_seizs_per_sess_dict[dkey] = cnt_seiz_per_sess
    print(f"Session {cnt_tot_sess:04d}/{1645}", end = "   ")
    print(f"ID: {dkey}", end = "\t")
    print(f"Total Records: {cnt_recs_per_sess:2d}", end = "\t")
    print(f"Total Seizures: {cnt_seiz_per_sess:3d}", end = "\t")
    #print(f"Cumulative Time (s): {cum_tot_time}")
    print(f"Cumulative Time: {cum_tot_time/60:8.2f} mins", end = "\r")
    time.sleep(0.1)
seiz_cum_strt_time_list = list(filter(None, seiz_cum_strt_time_list))
seiz_cum_ends_time_list = list(filter(None, seiz_cum_ends_time_list))

num_of_sess_no_seiz = cnt_tot_sess-len(cnt_sess_with_seiz_list)
num_of_sess_wi_seiz = len(cnt_sess_with_seiz_list)
assert num_of_sess_no_seiz == len(pid_sess_zero_seiz_list)
print(f'There are {num_of_sess_no_seiz} sessions with no seizures and {num_of_sess_wi_seiz} sessions with seizures.')
print("The lengths of the list of patients with seizures, cumulative seizure start, seizure end and total record times:")
print(len(seiz_cum_strt_time_list), len(seiz_cum_ends_time_list), 
      len(seiz_cum_ends_time_list), len(cum_time_with_seiz_list))

'''
This cell explains how the gap times are calculated for each dictionary including training, validation and testing.
$SPH$ = 30
$SOP$ = 2
$GAP_{min}$ = $SOP + SPH$
'''

count_seizures_gap_time_dict = {'trn':[], 'vld':[], 'tst':[]}
tid = 'trn'
for tid in ['trn', 'vld', 'tst']:
    for x_list, y_list in zip(seiz_cum_strt_time_dict[tid], seiz_cum_ends_time_dict[tid]):
        for i in range(len(x_list)):
            if i == 0:
                act_gap_time = x_list[i] - 0
            else:
                act_gap_time = x_list[i] - y_list[i-1]
            count_seizures_gap_time_dict[tid].append(act_gap_time)

    print("The gap times of the first 10 seizures in the", tid, "set (minutes):")
    for x in count_seizures_gap_time_dict[tid][:16]:
        print(f'{x/60:04.2f}', end  = "\t")
    print()
    
trn_ids_count = 0
vld_ids_count = 0
tst_ids_count = 0
trn_pat_ids_list = []
vld_pat_ids_list = []
tst_pat_ids_list = []
all_pat_ids_dict = {'trn':[], 'vld':[], 'tst':[]}
seizure_pat_ids_dict = {'trn':[], 'vld':[], 'tst':[]}
nonseiz_pat_ids_dict = {'trn':[], 'vld':[], 'tst':[]}
seizure_ssn_ids_dict = {'trn':[], 'vld':[], 'tst':[]}
nonseiz_ssn_ids_dict = {'trn':[], 'vld':[], 'tst':[]}
all_seizure_pat_ids_list = []
all_pat_session_cnt_dict = {'trn':{}, 'vld':{}, 'tst':{}}
all_nonseiz_pat_ids_list = []

for pid in pat_ids_lst:
    all_pat_ids_dict[pid[:3]].append(pid) 
    dst_typ = pid[:3]
    pid_val = pid[4:]
    all_pat_session_cnt_dict[dst_typ][pid_val] = []
    for x in list_of_keys:
        if pid_val == x[4:7]:
            all_pat_session_cnt_dict[dst_typ][pid_val].append(x)    
    if dst_typ == 'trn':
        trn_ids_count += 1
        trn_pat_ids_list.append(pid)
    if dst_typ == 'vld':
        vld_ids_count += 1
        vld_pat_ids_list.append(pid)
    if dst_typ == 'tst':
        tst_ids_count += 1
        tst_pat_ids_list.append(pid)
        
tot_ids_count = trn_ids_count + vld_ids_count + tst_ids_count
tot_ids_dict_length = len(all_pat_ids_dict['trn']) + len(all_pat_ids_dict['vld']) + len(all_pat_ids_dict['tst'])

assert len(all_pat_ids_dict['trn']) == trn_ids_count
assert len(all_pat_ids_dict['vld']) == vld_ids_count
assert len(all_pat_ids_dict['tst']) == tst_ids_count
assert tot_ids_dict_length == tot_ids_count

trn_ssn_count = 0
vld_ssn_count = 0
tst_ssn_count = 0

trn_tot_seizures_cnt = 0
tst_tot_seizures_cnt = 0
vld_tot_seizures_cnt = 0
tot_tot_seizures_cnt = 0
trn_ssns_with_seizures_cnt = 0
trn_ssns_zero_seizures_cnt = 0
vld_ssns_with_seizures_cnt = 0
vld_ssns_zero_seizures_cnt = 0
tst_ssns_with_seizures_cnt = 0
tst_ssns_zero_seizures_cnt = 0

trn_pats_with_seizures_cnt = 0
trn_pats_zero_seizures_cnt = 0
vld_pats_with_seizures_cnt = 0
vld_pats_zero_seizures_cnt = 0
tst_pats_with_seizures_cnt = 0
tst_pats_zero_seizures_cnt = 0

trn_pat_session_cnt_list = []
trn_pat_session_cnt_dict = {}
trn_pat_session_ids_dict = {}
vld_pat_session_cnt_list = []
vld_pat_session_cnt_dict = {}
vld_pat_session_ids_dict = {}
tst_pat_session_cnt_list = []
tst_pat_session_cnt_dict = {}
tst_pat_session_ids_dict = {}

for pid in trn_pat_ids_list:
    temp_session_count = 0
    temp_count_seiz_sess_per_pat = 0
    trn_pat_session_ids_dict[pid] = []
    for sess_id in list_of_keys:
        if pid[4:] == sess_id[4:7]:
            temp_session_count += 1
            trn_pat_session_ids_dict[pid].append(sess_id)
            if sess_id in pid_sess_with_seiz_list:
                temp_count_seiz_sess_per_pat += 1
                trn_ssns_with_seizures_cnt += 1
                trn_tot_seizures_cnt += cnt_seizs_per_sess_dict[sess_id]
            else:
                trn_ssns_zero_seizures_cnt += 1
    trn_pat_session_cnt_list.append(temp_session_count)
    trn_pat_session_cnt_dict[pid] = temp_session_count
    trn_ssn_count += temp_session_count
    if temp_count_seiz_sess_per_pat > 0:
        trn_pats_with_seizures_cnt += 1
        seizure_pat_ids_dict[pid[:3]].append(pid)
        all_seizure_pat_ids_list.append(pid)
    else:
        trn_pats_zero_seizures_cnt += 1
        nonseiz_pat_ids_dict[pid[:3]].append(pid)
        all_nonseiz_pat_ids_list.append(pid)
        
for pid in vld_pat_ids_list:
    temp_session_count = 0
    temp_count_seiz_sess_per_pat = 0
    vld_pat_session_ids_dict[pid] = []
    for sess_id in list_of_keys:
        if pid[4:] == sess_id[4:7]:
            temp_session_count += 1
            vld_pat_session_ids_dict[pid].append(sess_id)
            if sess_id in pid_sess_with_seiz_list:
                temp_count_seiz_sess_per_pat += 1
                vld_ssns_with_seizures_cnt += 1
                vld_tot_seizures_cnt += cnt_seizs_per_sess_dict[sess_id]
            else:
                vld_ssns_zero_seizures_cnt += 1
    vld_pat_session_cnt_list.append(temp_session_count)
    vld_pat_session_cnt_dict[pid] = temp_session_count
    vld_ssn_count += temp_session_count
    if temp_count_seiz_sess_per_pat > 0:
        vld_pats_with_seizures_cnt += 1
        seizure_pat_ids_dict[pid[:3]].append(pid)
        all_seizure_pat_ids_list.append(pid)
    else:
        vld_pats_zero_seizures_cnt += 1
        nonseiz_pat_ids_dict[pid[:3]].append(pid)
        all_nonseiz_pat_ids_list.append(pid)
        
for pid in tst_pat_ids_list:
    temp_session_count = 0
    temp_count_seiz_sess_per_pat = 0
    tst_pat_session_ids_dict[pid] = []
    for sess_id in list_of_keys:
        if pid[4:] == sess_id[4:7]:
            temp_session_count += 1
            tst_pat_session_ids_dict[pid].append(sess_id)
            if sess_id in pid_sess_with_seiz_list:
                temp_count_seiz_sess_per_pat += 1
                tst_ssns_with_seizures_cnt += 1
                tst_tot_seizures_cnt += cnt_seizs_per_sess_dict[sess_id]
            else:
                tst_ssns_zero_seizures_cnt += 1
    tst_pat_session_cnt_list.append(temp_session_count)
    tst_pat_session_cnt_dict[pid] = temp_session_count
    tst_ssn_count += temp_session_count
    if temp_count_seiz_sess_per_pat > 0:
        tst_pats_with_seizures_cnt += 1
        seizure_pat_ids_dict[pid[:3]].append(pid)
        all_seizure_pat_ids_list.append(pid)
    else:
        tst_pats_zero_seizures_cnt += 1
        nonseiz_pat_ids_dict[pid[:3]].append(pid)
        all_nonseiz_pat_ids_list.append(pid)
        
pats_with_seizures_seizure_session_ids_dict = {}
pats_with_seizures_nonseiz_session_ids_dict = {}
for x in all_seizure_pat_ids_list:
    pats_with_seizures_seizure_session_ids_dict[x] = []
    pats_with_seizures_nonseiz_session_ids_dict[x] = []
for x in pid_sess_with_seiz_list:
    temp_key = x[:7]
    if temp_key in all_seizure_pat_ids_list:
        pats_with_seizures_seizure_session_ids_dict[temp_key].append(x)
measured_length = 0
for x in pats_with_seizures_seizure_session_ids_dict:
    measured_length += len(pats_with_seizures_seizure_session_ids_dict[x])
print(measured_length)
print(len(pid_sess_with_seiz_list))
assert len(pid_sess_with_seiz_list) == measured_length

for dkey in list_of_keys:
    temp_pat_key = dkey[:7]
    if temp_pat_key in all_seizure_pat_ids_list and dkey not in pats_with_seizures_seizure_session_ids_dict[temp_pat_key]:
        pats_with_seizures_nonseiz_session_ids_dict[temp_pat_key].append(dkey)
measured_length = 0
for x in pats_with_seizures_nonseiz_session_ids_dict:
    measured_length += len(pats_with_seizures_nonseiz_session_ids_dict[x])
print(measured_length)
pats_with_seizures_wthzero_noszssn_ids_list = []
for x in pats_with_seizures_nonseiz_session_ids_dict:
    if not pats_with_seizures_nonseiz_session_ids_dict[x]:
        pats_with_seizures_wthzero_noszssn_ids_list.append(x)
print(len(pats_with_seizures_wthzero_noszssn_ids_list))

benchmark_sph_list = [2, 5, 15, 30]
benchmark_sop_list = [1, 2, 5]
benchmark_tot_list = [(a, b) for a in benchmark_sph_list for b in benchmark_sop_list]
bnchmrk_names_list =[f"bmrk{1+i:02d}" for i in range(len(benchmark_tot_list))]
preferred_gap_times_mins_list = [x[0] + x[1] for x in benchmark_tot_list]
print(benchmark_tot_list)
print(preferred_gap_times_mins_list)
print(bnchmrk_names_list)
print(sorted(set(preferred_gap_times_mins_list))+[45, 60, 90, 120])

#******************************************************************************#
#--------------------- Sec. 10: Create the Interim Dataset ---------------------#
'''
In the following cells, we enumerate the benchmarks and use them to generate 
the prefixes for the file names. In the next stage, we then try to calculate 
the gap time and enumerate the seizures that belong to each benchmark. Once 
we have those seizures, we figure out the session ID of each seizure and assign 
each seizure a "number" either implicitly via list ordering or explicitly via 
a count. For that particular benchmark, we calculate the preictal and interictal 
size depending upon the sph, sop and sampling durations. Initially, we double 
the preictal size to account for the interictal samples in a balanced manner 
and create 3D matrix of zeros of size L x 17 x N, where L is the number of 
preictal + interictal samples and N is the number of samples in each element 
which would be the sampling rate multiplied by the $SPH$.  

We then focus on using the sampling rate dictionary to figure out what should 
be the data size for each initial intermediate array. Once we have the intermediate 
array, depending on the sampling rate, we either upsample or downsample to 256 Hz.

$F_s$ = 256 <br>
$T_w \in \{4, 5, 10, 20\}$ seconds <br>
$SPH \in \{2, 5, 15, 30\}$ minutes <br>
$SOP \in \{1, 2, 5\}$ minutes <br>
$L_{pre} = F_s \times SPH$ <br>
$L_{tot} = 2L_{pre}$ <br>
$D =  L_{chs} \times L_{tot}$ <br>
$S_i = \dfrac{60\times SPH}{T_w}$ for $i = 1, 2, \ldots, b$ 
and $b \in \mathcal{B} = \{1, \ldots, B\}$ where $B = 12$ <br>
'''

smp_rte_secs = 256
crss_val_fld = 0

f01_dset_name_strg = "tuhszr"

f02_dtyp_intr_strg = "interm"
f02_dtyp_sfld_strg = "sngfld"
f02_dtyp_mfld_strg = "mltfld"

f03_scld_stat_strg = "unscld"
f04_filt_stat_strg = "unfilt"
f05_blnc_stat_strg = "blcrnd"
f06_srat_strg_strg = "srate{:d}Hz".format(smp_rte_secs)

f12_tusz_strd_strg = "tuhstd"
f12_strt_strd_strg = "strtfd"
# f12_psid_nmbr_strg = "{:3s}{:3s}".format(patient_idno, session_idno)

f13_cval_fold_strg = "fold{0:2d}".format(crss_val_fld)
# f13_szrs_nmbr_strg = "szr{:03d}".format(seizure_nmbr)

f14_trns_labl_strg = 'train'
f14_vlds_labl_strg = 'valid'
f14_tsts_labl_strg = 'tests'

f15_vals_labl_strg = 'values'
f15_labs_labl_strg = 'labels'

f16_feat_hdf5_strg = 'hdf5'
f16_feat_nmpy_strg = 'npy'
f16_labs_frmt_strg = 'csv' 

prefix = (f"{f01_dset_name_strg}_{f02_dtyp_intr_strg}_{f03_scld_stat_strg}_"
          f"{f04_filt_stat_strg}_{f05_blnc_stat_strg}_{f06_srat_strg_strg}")
# benchmark_prefix_2 = f"_{f11_over_stat_strg}"

train_values_suffix = f"{f14_trns_labl_strg}_{f15_vals_labl_strg}.{f16_feat_nmpy_strg}"
train_labels_suffix = f"{f14_trns_labl_strg}_{f15_labs_labl_strg}.{f16_labs_frmt_strg}"
valid_values_suffix = f"{f14_vlds_labl_strg}_{f15_vals_labl_strg}.{f16_feat_nmpy_strg}"
valid_labels_suffix = f"{f14_vlds_labl_strg}_{f15_labs_labl_strg}.{f16_labs_frmt_strg}"
tests_values_suffix = f"{f14_tsts_labl_strg}_{f15_vals_labl_strg}.{f16_feat_nmpy_strg}"
tests_labels_suffix = f"{f14_tsts_labl_strg}_{f15_labs_labl_strg}.{f16_labs_frmt_strg}"

values_suffix = f"{f15_vals_labl_strg}.{f16_feat_nmpy_strg}"
labels_suffix = f"{f15_labs_labl_strg}.{f16_labs_frmt_strg}"

print(prefix)
print()
print(train_values_suffix, train_labels_suffix)
print(valid_values_suffix, valid_labels_suffix)
print(tests_values_suffix,tests_labels_suffix)
print()
print(values_suffix)
print(labels_suffix)

win_len_secs = 5
ovr_len_secs = 0
crss_val_fld = 0
num_of_bmrks = len(bnchmrk_names_list)
f01_dset_name_strg = "tuhszr"

f10_segs_lens_strg = "seg{:02d}s".format(win_len_secs)
f11_over_stat_strg = "ovr{:02d}s".format(ovr_len_secs)

npy_file_dirs_dict = {}
cnt_dset_bmrk_dict = {"trn": num_of_bmrks*[0], "vld": num_of_bmrks*[0], "tst": num_of_bmrks*[0]}
cnt_tots_bmrk_list = num_of_bmrks*[0]
seiz_cum_strt_time_sph_mins_dict = {}
pres_cum_strt_time_sph_mins_dict = {}
pres_cum_strt_samp_sph_mins_dict = {}

for i in range(num_of_bmrks):
    benchmark_id = bnchmrk_names_list[i]
    npy_file_dirs_dict[benchmark_id] = []
    sph_len_mins = benchmark_tot_list[i][0]
    sop_len_mins = benchmark_tot_list[i][1]
    gap_len_mins = sph_len_mins + sop_len_mins
    f07_bmrk_idnm_strg = benchmark_id # "bmrk{:02d}".format(benchmark_id)
    f08_spht_secs_strg = "sph{:02d}s".format(sph_len_mins)
    f09_sopt_secs_strg = "sop{:02d}s".format(sop_len_mins)
    benchmark_prefix = (f"{f07_bmrk_idnm_strg}_{f08_spht_secs_strg}_{f09_sopt_secs_strg}_"
                        f"{f10_segs_lens_strg}_{f11_over_stat_strg}")
    print(f"{prefix}_{benchmark_prefix}", end = "\t")
    seiz_cum_strt_time_sph_mins_list = []
    pres_cum_strt_time_sph_mins_list = []
    pres_cum_strt_samp_sph_mins_list = []
    npy_file_dirs_list = []
    for dkey in list_of_keys:
        for x_list, y_list in zip(seiz_cum_strt_time_dict_unsorted[dkey], seiz_cum_ends_time_dict_unsorted[dkey]):
            for j in range(len(x_list)):
                if j == 0:
                    act_gap_time = x_list[j] - 0
                else:
                    act_gap_time = x_list[j] - y_list[j-1]
                if act_gap_time >= gap_len_mins*60:
                    # cur_samp_rate = sampling_rate_dict[dkey][0]
                    seiz_cum_strt_time_sph_mins_list.append(x_list[j])
                    pres_cum_strt_temp_sph_mins = x_list[j]-(gap_len_mins*60)
                    if pres_cum_strt_temp_sph_mins < 0:
                        pres_cum_strt_time_sph_mins_list.append(0)
                    else:
                        pres_cum_strt_time_sph_mins_list.append(pres_cum_strt_temp_sph_mins)
                    pres_cum_strt_samp_sph_mins_list.append(int(pres_cum_strt_time_sph_mins_list[-1]*256))
                    # pres_cum_strt_time_sph_mins_list.append(pres_cum_strt_temp_sph_mins)
                    # pres_cum_strt_samp_sph_mins_list.append(int(pres_cum_strt_time_sph_mins_list[-1]*cur_samp_rate))
                    npy_file_dirs_list.append(f'{mont_path}{dkey}.npy')
                    cnt_dset_bmrk_dict[dkey[:3]][i] += 1 
                    cnt_tots_bmrk_list[i] += 1                   
                # print(npy_file_dirs_list[-1])
    npy_file_dirs_dict[benchmark_id] = npy_file_dirs_list
    seiz_cum_strt_time_sph_mins_dict[benchmark_id] = seiz_cum_strt_time_sph_mins_list
    pres_cum_strt_time_sph_mins_dict[benchmark_id] = pres_cum_strt_time_sph_mins_list
    pres_cum_strt_samp_sph_mins_dict[benchmark_id] = pres_cum_strt_samp_sph_mins_list
    print(len(npy_file_dirs_dict[benchmark_id]))
    # print(benchmark_prefix)
    
train_valid_or_test_dict = {"trn": "train", "vld": "valid", "tst": "tests"}
cnt_vals_bmrk_dict = {"trn": num_of_bmrks*[0], "vld": num_of_bmrks*[0], "tst": num_of_bmrks*[0]}
tot_sng_fld_size_list = num_of_bmrks*[0]
for i in range(len(bnchmrk_names_list)-1, -1, -1):
#for i in range(11, 9, -1):
    benchmark_id = bnchmrk_names_list[i]
    npy_file_dirs_list = npy_file_dirs_dict[benchmark_id] 
    sph_len_mins = benchmark_tot_list[i][0]
    sop_len_mins = benchmark_tot_list[i][1]
    gap_len_mins = sph_len_mins + sop_len_mins
    
    f07_bmrk_idnm_strg = benchmark_id # "bmrk{:02d}".format(benchmark_id)
    f08_spht_secs_strg = "sph{:02d}m".format(sph_len_mins)
    f09_sopt_secs_strg = "sop{:02d}m".format(sop_len_mins)
    benchmark_prefix = (f"{f07_bmrk_idnm_strg}_{f08_spht_secs_strg}_{f09_sopt_secs_strg}_"
                        f"{f10_segs_lens_strg}_{f11_over_stat_strg}")
    cnt_npy_file_dirs_list = 0
    for j in range(len(npy_file_dirs_list)):
        file_path = npy_file_dirs_list[j]
        dataset_idno = npy_file_dirs_list[j].split('/')[-1][:3]
        patient_idno = file_path[-16:-13]
        session_idno = file_path[-11:-8]
        seizure_nmbr = j+1
        f12_psid_nmbr_strg = "{:3s}{:3s}".format(patient_idno, session_idno)
        f13_szrs_nmbr_strg = "sz{:04d}".format(seizure_nmbr)
        f14_dset_labl_strg = train_valid_or_test_dict[file_path[-20:-17]]
        #smp_rte_secs
        save_file_name = (f"{prefix}_{benchmark_prefix}_{f13_szrs_nmbr_strg}_{f12_psid_nmbr_strg}_"
                          f"{f14_dset_labl_strg}_{values_suffix}")
        save_file_path = f"{intr_path}{save_file_name}"
        num_of_pre_segs = sph_len_mins*60//win_len_secs
        num_of_tot_segs = 2*num_of_pre_segs
        num_of_seg_samps = win_len_secs*smp_rte_secs
        win_seg_samps = win_len_secs*smp_rte_secs
        #temp_array_read = np.load(file_path)
        #print(f"Processing BM{i+1:02d} SSN {j+1:04d}/{len(npy_file_dirs_list):04d} with data shape"
        #      f" {temp_array_read.shape[0]:2d} x {temp_array_read.shape[1]:<12d}", end ="\r")
        print(f"Processing BM{i+1:02d} SSN {j+1:04d}/{len(npy_file_dirs_list):04d} ", end = "\r")
        # temp_array_3D = np.zeros((num_of_tot_segs, 20, win_seg_samps))
        # temp_array_3D = 0.0001*np.random.rand(num_of_tot_segs, 20, win_seg_samps)
        if j==0:
            pass
        temp_key = file_path[-20:-13]
        temp_nsz = pats_with_seizures_nonseiz_session_ids_dict[temp_key]
        if temp_nsz:
            temp_npy = mont_path+temp_nsz[0]+'.npy'
            temp_mnt = np.load(temp_npy)
            temp_num = temp_mnt.shape[1]//win_seg_samps
            if num_of_pre_segs > 0 and temp_num > num_of_pre_segs:
                temp_array_read = np.load(file_path)
                print(f"Processing BM{i+1:02d} SSN {j+1:04d}/{len(npy_file_dirs_list):04d} with data shape"
                      f" {temp_array_read.shape[0]:2d} x {temp_array_read.shape[1]:<12d}", end ="\r")
                temp_array_3D = 10000*np.random.rand(num_of_tot_segs, 20, win_seg_samps)
                cnt_vals_bmrk_dict[dataset_idno][i] += 1
                cnt_npy_file_dirs_list += 1
                end_sequence = num_of_pre_segs
                for k in range(0,end_sequence):
                    start_ind = k*win_seg_samps
                    end_index = start_ind+win_seg_samps
                    temp_array_3D[k,:,:] = temp_mnt[:,start_ind:end_index]
                for k in range(num_of_pre_segs, num_of_tot_segs):
                    # start_ind = num_of_pre_segs-k+pres_cum_strt_samp_sph_mins_dict[benchmark_id][j]
                    start_ind = pres_cum_strt_samp_sph_mins_dict[benchmark_id][j] + (k-num_of_pre_segs)*win_seg_samps
                    end_index =  start_ind + win_seg_samps
                    temp_array_3D[k,:,:] = temp_array_read[:,start_ind:end_index]
                np.save(save_file_path, temp_array_3D)
            else:
                pass
                #print(f" 00 x {0:<12d}. Nothing to read!", end ="\r")
                #time.sleep(10000)
    tot_sng_fld_size_list[i] = num_of_tot_segs*cnt_npy_file_dirs_list
    print()

print('bmk: [',end = "")
for i in range(len(tot_sng_fld_size_list)-1):
    print(f'B{i+1:02d}, ', end = "")
print('B12]')
for dkey in cnt_vals_bmrk_dict:
    print(dkey,'\b: [', end = "")
    for x in cnt_vals_bmrk_dict[dkey][:-1]:
        print(f'{x:3d}', end = ", ")
    print(f'{cnt_vals_bmrk_dict[dkey][-1]:3d}]')
x_den_list = [48, 120, 360, 720]
count_dens = 0
print('tot: [', end = "")
for x in tot_sng_fld_size_list[:-1]:
    print(f'{x//x_den_list[count_dens//3]:3d}, ', end = "")
    count_dens += 1
print(f'{tot_sng_fld_size_list[-1]//x_den_list[-1]:3d}]')
print('Sizes:',tot_sng_fld_size_list)

#******************************************************************************#
#-------------- Sec. 11: Create the Single-Fold ML-Ready Dataset --------------#
'''
In the following cells, we enumerate the benchmarks and use them to generate 
the prefixes for the file names. In the next stage, we then try to calculate 
the gap time and enumerate the seizures that belong to each benchmark.

Sampling rate $F_s$ = 256 <br>
Window length (segment duration) $T_w \in \{4, 5, 10, 20\}$ seconds <br>
$SPH \in \{2, 5, 15, 30\}$ minutes <br>
$SOP \in \{1, 2, 5\}$ minutes <br>
Number of channels $C = 20$ <br>
Total samples in every window $L = F_s \times T_w$ <br>
Size of 2d EEG data for every window $D =  C \times L$ <br>
Number of preictal samples associated with each seizure 
$S_{pre} = \dfrac{60\times SPH}{T_w}$ <br>
Number of interictal samples are equal to the number 
of preictal samples in the balanced version. 
Therefore $S_{int} = S_{pre}$. <br>
Total 2d elements in each seizure in the $i^{th}$ benchmark 
$S_i = \dfrac{2\times 60\times SPH}{T_w}$ <br>
Each benchmark $i$ for $i = 1, 2, \ldots, B$ where $B = 12$ for our case. <br>

In each benchmark $i$, there are a total of $S_i$ data points, preictal and interictal 
combined for each seizure and each $i^{th}$ data point contains a 2d element of size $D$. <br>
For each $i^{th}$ benchmark, there will be $N_{trn,i}$, $N_{vld,i}$ and $N_{tst,i}$ 
training, validation and test seizures, respectively.  <br>
Total 2d elements needed to be stored in each single fold will then be: <br>
Training: $N_{trn,i}\times S_i$ <br>
Validation: $N_{vld,i}\times S_i$ <br>
Testing: $N_{tst,i}\times S_i$ <br>
'''

f01_dset_name_strg = "tuhszr"

f02_dtyp_intr_strg = "interm"
f02_dtyp_sfld_strg = "sngfld"
f02_dtyp_mfld_strg = "mltfld"

f12_tusz_strd_strg = "tuhstd"
f12_strt_strd_strg = "strtfd"
f12_psid_nmbr_strg = "{:3s}{:3s}".format(patient_idno, session_idno)

f13_cval_fold_strg = "fold{0:2d}".format(crss_val_fld)
f13_szrs_nmbr_strg = "szr{:03d}".format(seizure_nmbr)

prefix = (f"{f01_dset_name_strg}_{f02_dtyp_sfld_strg}_{f03_scld_stat_strg}_"
          f"{f04_filt_stat_strg}_{f05_blnc_stat_strg}_{f06_srat_strg_strg}")

train_values_suffix = f"{f14_trns_labl_strg}_{f15_vals_labl_strg}.{f16_feat_hdf5_strg}"
train_labels_suffix = f"{f14_trns_labl_strg}_{f15_labs_labl_strg}.{f16_labs_frmt_strg}"
valid_values_suffix = f"{f14_vlds_labl_strg}_{f15_vals_labl_strg}.{f16_feat_hdf5_strg}"
valid_labels_suffix = f"{f14_vlds_labl_strg}_{f15_labs_labl_strg}.{f16_labs_frmt_strg}"
tests_values_suffix = f"{f14_tsts_labl_strg}_{f15_vals_labl_strg}.{f16_feat_hdf5_strg}"
tests_labels_suffix = f"{f14_tsts_labl_strg}_{f15_labs_labl_strg}.{f16_labs_frmt_strg}"

values_suffix = f"{f15_vals_labl_strg}.{f16_feat_hdf5_strg}"
labels_suffix = f"{f15_labs_labl_strg}.{f16_labs_frmt_strg}"

print(prefix)
print()
print(train_values_suffix, train_labels_suffix)
print(valid_values_suffix, valid_labels_suffix)
print(tests_values_suffix,tests_labels_suffix)
print()
print(values_suffix)
print(labels_suffix)
          
smp_rte_secs = 256
win_len_secs = 5
num_of_chans = 20
smps_per_win = win_len_secs*smp_rte_secs
smps_2d_size = num_of_chans*smps_per_win
ovr_len_secs = 0
crss_val_fld = 0
f13_cval_fold_strg = "fold{:02d}".format(crss_val_fld)
path_list_all_bmarks = os.listdir(intr_path)
for i in range(len(bnchmrk_names_list)-1, -1, -1):
#for i in range(11, 9, -1):
    benchmark_id = bnchmrk_names_list[i]
    tot_sng_fld_size = tot_sng_fld_size_list[i]
    n_trn_i = cnt_vals_bmrk_dict["trn"][i]
    n_vld_i = cnt_vals_bmrk_dict["vld"][i]
    n_tst_i = cnt_vals_bmrk_dict["tst"][i]
     
    sph_len_mins = benchmark_tot_list[i][0]
    sop_len_mins = benchmark_tot_list[i][1]
    gap_len_mins = sph_len_mins + sop_len_mins
    
    s_i = 2*60*sph_len_mins//win_len_secs
    si_by_two = s_i//2
    train_fold_depth = n_trn_i*s_i
    valid_fold_depth = n_vld_i*s_i
    tests_fold_depth = n_tst_i*s_i
    
    temp_array_train = np.zeros((train_fold_depth, smps_per_win, num_of_chans))
    temp_array_valid = np.zeros((valid_fold_depth, smps_per_win, num_of_chans))
    temp_array_tests = np.zeros((tests_fold_depth, smps_per_win, num_of_chans))
    temp_labls_train = np.zeros(train_fold_depth)
    temp_labls_valid = np.zeros(valid_fold_depth)
    temp_labls_tests = np.zeros(tests_fold_depth)
    
    count_dset_type = {"trn": 0, "vld": 0, "tst": 0}
    count_train = 0
    count_valid = 0
    count_tests = 0
    
    for f in path_list_all_bmarks:
        if f.split('_')[6] == benchmark_id:
            temp_array_read = np.load(intr_path+f)
            dset_lab = f.split('_')[-2]
            dset_key = list(train_valid_or_test_dict.keys())[list(train_valid_or_test_dict.values()).index(dset_lab)]
            start_index = count_dset_type[dset_key]*s_i
            end_index = start_index + s_i
            if count_dset_type[dset_key] == 5:
                print(temp_array_read.shape, start_index, end_index, end = " ")
            count_dset_type[dset_key] += 1
            if dset_key == "trn":
                temp_array_train[start_index:end_index,:,:] = np.reshape(temp_array_read,(s_i, smps_per_win, num_of_chans))
                temp_labls_train[start_index:start_index+si_by_two] = 1
            if dset_key == "vld":
                temp_array_valid[start_index:end_index,:,:] = np.reshape(temp_array_read,(s_i, smps_per_win, num_of_chans))
                temp_labls_valid[start_index:start_index+si_by_two] = 1
            if dset_key == "tst":
                temp_array_tests[start_index:end_index,:,:] = np.reshape(temp_array_read,(s_i, smps_per_win, num_of_chans))
                temp_labls_tests[start_index:start_index+si_by_two] = 1
    print()
    try:
        assert (n_trn_i == count_dset_type["trn"] and n_vld_i == count_dset_type["vld"] 
                and n_tst_i == count_dset_type["tst"])
    except AssertionError:
        print("Failed one or more assertions")
    
    f07_bmrk_idnm_strg = benchmark_id # "bmrk{:02d}".format(benchmark_id)
    f08_spht_secs_strg = "sph{:02d}m".format(sph_len_mins)
    f09_sopt_secs_strg = "sop{:02d}m".format(sop_len_mins)
    benchmark_prefix = (f"{f07_bmrk_idnm_strg}_{f08_spht_secs_strg}_{f09_sopt_secs_strg}_"
                        f"{f10_segs_lens_strg}_{f11_over_stat_strg}_{f13_cval_fold_strg}_{f12_tusz_strd_strg}")
    print(f"{prefix}_{benchmark_prefix}_{train_values_suffix}")
    print(f"{prefix}_{benchmark_prefix}_{train_labels_suffix}")
    print(f"{prefix}_{benchmark_prefix}_{valid_values_suffix}")
    print(f"{prefix}_{benchmark_prefix}_{valid_labels_suffix}")
    print(f"{prefix}_{benchmark_prefix}_{tests_values_suffix}")
    print(f"{prefix}_{benchmark_prefix}_{tests_labels_suffix}")
    print(temp_array_train.shape, temp_array_valid.shape, temp_array_tests.shape)
    print(count_dset_type)
    print()
    
    X_trn_hdf = h5py.File(f"{sf00_path}{prefix}_{benchmark_prefix}_{train_values_suffix}",'w')
    X_trn_hdf.create_dataset('tracings', data = temp_array_train)
    X_trn_hdf.close()  
    
    with open(f"{sf00_path}{prefix}_{benchmark_prefix}_{train_labels_suffix}", 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        A = []
        for x in temp_labls_train.astype(int):
            A. append([x])
        #print(A)
        csvwriter.writerows(A)
    
    X_vld_hdf = h5py.File(f"{sf00_path}{prefix}_{benchmark_prefix}_{valid_values_suffix}",'w')
    X_vld_hdf.create_dataset('tracings', data = temp_array_valid)
    X_vld_hdf.close()  
    
    with open(f"{sf00_path}{prefix}_{benchmark_prefix}_{valid_labels_suffix}", 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        A = []
        for x in temp_labls_valid.astype(int):
            A. append([x])
        #print(A)
        csvwriter.writerows(A)
    
    X_tst_hdf = h5py.File(f"{sf00_path}{prefix}_{benchmark_prefix}_{tests_values_suffix}",'w')
    X_tst_hdf.create_dataset('tracings', data = temp_array_tests)
    X_tst_hdf.close()  
    
    with open(f"{sf00_path}{prefix}_{benchmark_prefix}_{tests_labels_suffix}", 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        A = []
        for x in temp_labls_tests.astype(int):
            A. append([x])
        #print(A)
        csvwriter.writerows(A)