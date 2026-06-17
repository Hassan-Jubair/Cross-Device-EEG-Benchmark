# harmonization_pipeline.py
import os
import numpy as np
import mne
from moabb.datasets import Cho2017, PhysionetMI, Lee2019_MI, Shin2017A, Weibo2014
from moabb.paradigms import MotorImagery
import scipy.linalg as la
import warnings
from tqdm.auto import tqdm

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

warnings.filterwarnings('ignore')

datasets = [PhysionetMI(), Cho2017(), Lee2019_MI(), Shin2017A(accept=True), Weibo2014()]
paradigm = MotorImagery(n_classes=2, resample=100, fmin=8, fmax=32)

target_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T7', 'C3', 'Cz', 
                   'C4', 'T8', 'P7', 'P3', 'Pz', 'P4', 'P8', 'O1', 'Oz', 'O2']

montage = mne.channels.make_standard_montage('standard_1020')

def euclidean_alignment(X):
    covariances = np.array([np.cov(x) for x in X])
    mean_cov = np.mean(covariances, axis=0)
    
    mean_cov += np.eye(mean_cov.shape[0]) * 1e-6
    eigvals, eigvecs = la.eigh(mean_cov)
    eigvals = np.clip(eigvals, a_min=1e-8, a_max=None)
    
    isqrt_cov = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
    X_aligned = np.array([isqrt_cov @ x for x in X])
    return X_aligned

def harmonize_channels(raw, target_channels, montage):
    target_clean = [ch.upper().strip() for ch in target_channels]
    ch_names_clean = [ch.upper().replace('.', '').strip() for ch in raw.ch_names]
    
    rename_dict = {}
    for orig, up in zip(raw.ch_names, ch_names_clean):
        if up in target_clean:
            idx = target_clean.index(up)
            rename_dict[orig] = target_channels[idx]
    
    try:
        raw.rename_channels(rename_dict)
    except Exception:
        pass 
        
    raw.pick([ch for ch in raw.ch_names if ch in target_channels])
    missing_channels = [ch for ch in target_channels if ch not in raw.ch_names]
    
    if missing_channels:
        info = mne.create_info(ch_names=missing_channels, sfreq=raw.info['sfreq'], ch_types='eeg')
        flat_data = np.zeros((len(missing_channels), raw.n_times))
        raw_missing = mne.io.RawArray(flat_data, info, verbose=False)
        raw.add_channels([raw_missing], force_update_info=True)
        raw.info['bads'].extend(missing_channels)
        
    raw.set_montage(montage, match_case=False, on_missing='ignore')
    
    if len(raw.info['bads']) > 0:
        raw.interpolate_bads(reset_bads=True, verbose=False)
        
    raw.reorder_channels(target_channels)
    return raw

final_X, final_y, final_domains = [], [], []

for domain_idx, dataset in enumerate(datasets):
    checkpoint_file = f"checkpoint_{dataset.code}.npz"
    
    if os.path.exists(checkpoint_file):
        print(f"Loading checkpoint for {dataset.code}")
        data = np.load(checkpoint_file)
        final_X.append(data['X'])
        final_y.append(data['y'])
        final_domains.append(data['domains'])
        continue

    print(f"Processing: {dataset.code}")
    
    subjects = dataset.subject_list
    X, labels, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)
    X = X[:, :, :300]
    
    try:
        subj_data = dataset.get_data(subjects=[subjects[0]])
        subj_id = subjects[0]
        sess_id = list(subj_data[subj_id].keys())[0]
        run_id = list(subj_data[subj_id][sess_id].keys())[0]
        raw_sample = subj_data[subj_id][sess_id][run_id].copy()
        raw_sample.pick_types(eeg=True)
        ch_names = raw_sample.ch_names[:X.shape[1]]
    except Exception as e:
        if dataset.code == 'Shin2017A':
            fallback_channels = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10']
            ch_names = fallback_channels[:X.shape[1]]
        else:
            ch_names = [str(c) for c in range(X.shape[1])]
        
    X_interpolated = []
    
    for i in tqdm(range(len(X)), desc="Interpolating Channels"):
        info = mne.create_info(ch_names=ch_names, sfreq=100, ch_types='eeg')
        raw = mne.io.RawArray(X[i], info, verbose=False)
        raw_harmonized = harmonize_channels(raw, target_channels, montage)
        X_interpolated.append(raw_harmonized.get_data())
        
    X_interpolated = np.array(X_interpolated)
    X_aligned = euclidean_alignment(X_interpolated)
    
    unique_labels = np.unique(labels)
    y_bin = np.where(labels == unique_labels[0], 0, 1)
    domain_array = np.full(len(labels), domain_idx)
    
    np.savez_compressed(checkpoint_file, X=X_aligned, y=y_bin, domains=domain_array)
    
    final_X.append(X_aligned)
    final_y.append(y_bin)
    final_domains.append(domain_array)

X_final = np.concatenate(final_X, axis=0)
y_final = np.concatenate(final_y, axis=0)
domains_final = np.concatenate(final_domains, axis=0)

final_filename = 'cross_device_eeg_aligned_FINAL.npz'
np.savez_compressed(final_filename, X=X_final, y=y_final, domains=domains_final, channels=target_channels)
print(f"Dataset compiled. Shape: {X_final.shape}")

if IN_COLAB:
    files.download(final_filename)