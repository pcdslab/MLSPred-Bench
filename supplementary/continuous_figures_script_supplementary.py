import os
import re
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf

def load_models(models_dir, pattern):
    """Load .hdf5 models matching pattern from models_dir."""
    all_files  = os.listdir(models_dir)
    model_files = [
        fname for fname in all_files
        if pattern.match(fname)
    ]
    model_files.sort(key=lambda fn: int(pattern.match(fn).group(1)))
    models = []
    for fname in model_files:
        path = os.path.join(models_dir, fname)
        print(f'Loading {fname}…')
        models.append(load_model(path))
    return models

def parse_metadata(metadata_path):
    """Parse metadata .txt to cumulative seizure times."""
    seizure_times = []
    cumulative_off = 0.0
    prev_file_dur = None
    pat_file_start = re.compile(r'File Start Time:\s*([\d.]+)')
    pat_file_end   = re.compile(r'File End Time:\s*([\d.]+)')
    pat_s_start    = re.compile(r'Seizure Start Time:\s*([\d.]+)')
    pat_s_end      = re.compile(r'Seizure End Time:\s*([\d.]+)')
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith('File Name:'):
                if prev_file_dur is not None:
                    cumulative_off += prev_file_dur
                prev_file_dur = None
            m = pat_file_start.match(line)
            if m:
                file_start = float(m.group(1))
                continue
            m = pat_file_end.match(line)
            if m:
                file_end = float(m.group(1))
                prev_file_dur = file_end - file_start
                continue
            m = pat_s_start.match(line)
            if m:
                rel = float(m.group(1))
                seizure_times.append([cumulative_off + rel, None])
                continue
            m = pat_s_end.match(line)
            if m and seizure_times:
                rel = float(m.group(1))
                seizure_times[-1][1] = cumulative_off + rel
    return [(s,e) for s,e in seizure_times if e is not None]

def load_eeg(eeg_path, expected_channels=20):
    """Load .npy EEG, ensure shape (samples, channels)."""
    raw = np.load(eeg_path)
    if raw.ndim != 2:
        raise ValueError(f'Expected 2D array, got {raw.shape}')
    if raw.shape[0] == expected_channels and raw.shape[1] > expected_channels:
        return raw.T
    elif raw.shape[1] == expected_channels and raw.shape[0] > expected_channels:
        return raw
    else:
        raise ValueError('Cannot infer EEG orientation')

def segment_eeg(eeg, sampling_rate, segment_secs):
    """Segment EEG into non-overlapping windows and return segments, times, seg_times."""
    n_samples, _ = eeg.shape
    seg_len = sampling_rate * segment_secs
    n_segments = n_samples // seg_len
    segments = np.stack([eeg[i*seg_len:(i+1)*seg_len] 
                         for i in range(n_segments)], axis=0).astype(np.float32)
    times = np.arange(n_samples) / sampling_rate
    seg_times = (np.arange(n_segments)*seg_len + seg_len/2) / sampling_rate
    return segments, times, seg_times

def predict_probs(models, segments):
    """Run each model on segments; return probability array."""
    n_models = len(models)
    n_segments = segments.shape[0]
    probs = np.zeros((n_models, n_segments), dtype=np.float32)
    X = segments  # shape (batch, time, channels)
    for i, m in enumerate(models):
        #m.compile(run_eagerly=True)
        m.compile()
        preds = m.predict(X, verbose=0).flatten()
        probs[i] = preds
    tf.keras.backend.clear_session()
    return probs

def compute_firing_power(probs, window_size, threshold=0.5):
    n_models, n_segments = probs.shape
    cont = np.array([np.convolve(probs[i], 
                                  np.ones(window_size)/window_size, 
                                  mode='same') 
                     for i in range(n_models)])
    binarized = (probs >= threshold).astype(np.float32)
    binfp = np.array([np.convolve(binarized[i], 
                                  np.ones(window_size)/window_size, 
                                  mode='same') 
                      for i in range(n_models)])
    return cont, binfp

def plot_dual_axis(eeg, times, seg_times, seizure_times, probs, cont, binfp,
                   sampling_rate, segment_secs, model_idx, session_id, out_dir):
    """Plot stacked EEG + predictions (dual y-axis) for one benchmark."""
    eeg_uV = eeg * 1e6
    n_samples, n_channels = eeg_uV.shape
    p2p = eeg_uV.max() - eeg_uV.min()
    offset = p2p * 1.2
    x_marker = times[-1] - segment_secs
    risk_labels = {
        1/3: ' - Low risk',
        1/2: ' - Medium risk',
        2/3: ' - High risk'
    }
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax2 = ax.twinx()
    
    # 6) Plot stacked EEG on left axis
    for ch in range(n_channels):
        if ch == n_channels-1:
            ax.plot(times, eeg_uV[:, ch] + ch * offset, color='grey', linewidth=0.5, alpha=0.4, label='EEG Data')
        else:
            ax.plot(times, eeg_uV[:, ch] + ch * offset, color='grey', linewidth=0.5, alpha=0.4)

    # 7) Highlight seizure periods
    for start, end in seizure_times:
        ax.axvspan(start, end, color='red', alpha=0.3)

    # 8) Plot predictions on right axis
    ax2.plot(seg_times, probs[model_idx], color="#0000FF", label='Probability')
    #ax2.plot(seg_times, firing_power_cont[model_idx], color='cyan', label='Cont. FP')
    ax2.plot(seg_times, binfp[model_idx], color='red', linestyle='-.', label='Firing Power')

    # 9) Add small 'x' markers + in-plot risk labels on the right axis
    risk_labels = {
        1/3: ' - Low risk',
        1/2: ' - Medium risk',
        2/3: ' - High risk'
    }
    # x‐position for markers (a few seconds before your x‐axis max)
    x_marker = times[-1] - segment_secs

    for thr, col in [(1/3, 'green'), (1/2, 'orange'), (2/3, 'brown')]:
        # plot the x marker
        ax2.plot(
            x_marker, thr,
            color=col,
            linestyle='',
            marker='.',
            markersize=12,
            markeredgewidth=3#,
            #label=f'{thr:.2f} threshold'  # for your legend
        )
        # add the label text to the left of the marker
        ax2.text(
            x_marker + segment_secs,  # shift left
            thr,
            risk_labels[thr],
            color=col,
            va='center',
            ha='left',                   # right‐align text at its x‐position
            fontsize='large'
        )

    # 10) Axis limits
    ax.set_ylim(-0.5 * offset, (n_channels - 0.5) * offset)
    ax2.set_ylim(0, 1.01)

    # 11) Labels, title, legend
    ax.set_xlabel('Time (s)', fontsize='large')
    ax.set_ylabel('EEG (microvolts)', fontsize='large', loc = 'bottom')
    ax2.set_ylabel('Probability / Firing Power', fontsize='large', loc = 'bottom')
    ax.set_yticks([])

    ax.set_xlim([0, times[-1]])

    ax.set_title(f'Subject `{session_id}` EEG overlayed on prediction from BM{model_idx+1:02d}', fontsize='large')
    #handles, labels = ax2.get_legend_handles_labels()
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax2.legend(handles, labels, loc='upper right', ncol=1, fontsize='large')

    plt.tight_layout()
    plt.savefig(f'{out_dir}/single_{session_id}_BM{model_idx+1:02d}.jpeg', dpi=300)
    #plt.show()
    plt.close()

def plot_grid(eeg, times, seg_times, seizure_times, probs, cont, binfp,
              sampling_rate, segment_secs, out_dir, session_id):
    """Plot 4x3 grid of dual-axis plots for all benchmarks."""
    eeg_uV = eeg * 1e6
    n_samples, n_channels = eeg_uV.shape
    n_models, _ = probs.shape
    p2p = eeg_uV.max() - eeg_uV.min()
    offset = p2p * 1.2
    x_marker = times[-1] - segment_secs
    risk_labels = {
        1/3: ' - Low risk',
        1/2: ' - Medium risk',
        2/3: ' - High risk'
    }
    
    fig, axes = plt.subplots(4, 3, figsize=(24, 14), sharex=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i >= n_models:
            fig.delaxes(ax)
            continue

        ax2 = ax.twinx()
        # — stacked EEG on left axis —
        for ch in range(n_channels):
            if ch == n_channels-1:
                ax.plot(times, eeg_uV[:, ch] + ch * offset, color='grey', linewidth=0.5, alpha=0.4, label='EEG Data')
            else:
                ax.plot(times, eeg_uV[:, ch] + ch * offset, color='grey', linewidth=0.5, alpha=0.4)
        # — shade seizures —
        for start, end in seizure_times:
            ax.axvspan(start, end, color='red', alpha=0.2)

        # — predictions on right axis —
        ax2.plot(seg_times, probs[i],              color='blue',   label='Probability')
        #ax2.plot(seg_times, firing_power_cont[i],  color='cyan',   label='Cont FP')
        ax2.plot(seg_times, binfp[i],   color='red', linestyle='--', label='Firing Power')

        # — threshold lines on right axis —
        for thr, col in [(1/3, 'green'), (1/2, 'orange'), (2/3, 'brown')]:
            # plot the x marker
            ax2.plot(
                x_marker, thr,
                color=col,
                linestyle='',
                marker='.',
                markersize=12,
                markeredgewidth=3#,
                #label=f'{thr:.2f} threshold'  # for your legend
            )
            # add the label text to the left of the marker
            ax2.text(
                x_marker - segment_secs,  # shift left
                thr,
                risk_labels[thr][3:-5] + ' - ',
                color=col,
                va='center',
                ha='right',                   # right‐align text at its x‐position
                fontsize='large'
            )

        # — axis limits —
        ax.set_ylim(-0.5*offset, (n_channels - 0.5)*offset)
        ax2.set_ylim(0, 1)

        # — clean up & labels —
        ax.set_yticks([])           # hide EEG y-ticks
        if i % 3 == 0:
            ax.set_ylabel('EEG (microvolts)', fontsize='large')
        if i // 3 == 3:
            ax.set_xlabel('Time (s)', fontsize='large')
        if i % 3 == 2:
            ax2.set_ylabel('Prob / FP', fontsize='large')

        ax.set_title(f'Subject `{session_id}` BM{i+1:02d}', fontsize='large')
        ax.set_xlim([0, times[-1]])

    # 3) Global legend for predictions & thresholds
    #handles, labels = ax2.get_legend_handles_labels()
    handles = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize='large')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f'{out_dir}/multiple_{session_id}_all_BMs.jpeg', dpi=300)
    #plt.show()
    plt.close()
    return offset


if __name__ == '__main__':
    # user parameters       
    sampling_rate = 256
    segment_secs  = 5
    window_size   = 29
    models_dir    = 'path/to/model/checkpoints'
    model_pattern = re.compile(
        r"tuhszr_sngfld_unscld_unfilt_blcdet_srate256Hz_"
        r"bmrk(0[1-9]|1[0-2])_sph\d{2}m_sop\d{2}m_"
        r"seg05s_ovr00s_fold00_tuhstd_model_run00\.hdf5$"
    )
    models = load_models(models_dir, model_pattern)    
    out_dir = 'figures/output/directory'

    # pipeline
    offset_list = []
    met_dir = '/path/to/meta_data'
    eeg_dir = '/path/to/montage'
    dirs_list = os.listdir(met_dir)
    ids_list = []
    for d in dirs_list:
        ids_list.append(d[:-4])
    ids_list.sort(reverse=True)
    for session_id in ids_list[488:]:
        print(session_id)
        metadata_file = f'{met_dir}/{session_id}.txt' 
        eeg_path = f'{eeg_dir}/{session_id}.npy' 
        seizure_times = parse_metadata(metadata_file)
        eeg = load_eeg(eeg_path)
        segments, times, seg_times = segment_eeg(eeg, sampling_rate, segment_secs)
        probs = predict_probs(models, segments)
        cont, binfp = compute_firing_power(probs, window_size)
        
        # single-model plot
        try:
            plot_dual_axis(eeg, times, seg_times, seizure_times, probs, cont, binfp,
                       sampling_rate, segment_secs, model_idx=0,
                       session_id=session_id, out_dir=out_dir)
        except ValueError as e:
            print("Mismatch between the sizes of x and y.")
            print(e)
        except Exception as e:
            print("Something else went wrong.")
            print(e)
           
        #grid plot
        
        try:
            offset_temp = plot_grid(eeg, times, seg_times, seizure_times, probs, cont, binfp,
                  sampling_rate, segment_secs, out_dir, session_id)
        except ValueError as e:
            print("Mismatch between the sizes of x and y.")
            print(e)
        except Exception as e:
            print("Something else went wrong.")
            print(e)
        offset_list.append(offset_temp)
        #print(cont)
    print(offset_list)