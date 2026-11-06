import numpy as np
import mne

TARGET_CHANNELS = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T7', 'C3', 'Cz', 'C4', 'T8',
    'P7', 'P3', 'Pz', 'P4', 'P8',
    'O1', 'Oz', 'O2'
]
SHIN_MISSING_8 = ['Fp1', 'Fp2', 'Fz', 'C3', 'C4', 'O1', 'Oz', 'O2']
LEGACY_ALIAS_MAP = {'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8'}
MONTAGE_1005 = mne.channels.make_standard_montage('standard_1005')
MONTAGE_UPPER_MAP = {ch.upper(): ch for ch in MONTAGE_1005.ch_names}
for ch in TARGET_CHANNELS:
    MONTAGE_UPPER_MAP[ch.upper()] = ch

def clean_channel_names(orig_ch_names: list) -> list:
    cleaned, seen = [], set()
    for idx, ch in enumerate(orig_ch_names):
        c = ch.strip().replace('EEG ', '').replace('EEG-', '').replace('.', '').upper()
        c = LEGACY_ALIAS_MAP.get(c, c)
        canonical = MONTAGE_UPPER_MAP.get(c, ch.strip().replace('.', ''))
        if canonical in seen:
            canonical = f"{canonical}_dup{idx}"
        seen.add(canonical)
        cleaned.append(canonical)
    return cleaned

def harmonize_epochs_dense_to_20(X_raw: np.ndarray, orig_ch_names: list, sfreq: float = 100.0):
    """Dense-First spherical spline interpolation from native 10-05 montage prior to 20-channel CAR."""
    median_std = np.median(np.std(X_raw, axis=-1))
    X_scaled = X_raw * 1e6 if median_std < 1e-2 else X_raw.copy()
    cleaned_names = clean_channel_names(orig_ch_names)

    info = mne.create_info(ch_names=cleaned_names, sfreq=sfreq, ch_types='eeg')
    epochs = mne.EpochsArray(X_scaled, info, verbose=False)
    present_targets = [ch for ch in TARGET_CHANNELS if ch in epochs.ch_names]
    missing_targets = [ch for ch in TARGET_CHANNELS if ch not in epochs.ch_names]

    if missing_targets:
        missing_info = mne.create_info(ch_names=missing_targets, sfreq=sfreq, ch_types='eeg')
        zero_block = np.zeros((len(epochs), len(missing_targets), epochs.times.size), dtype=np.float64)
        epochs.add_channels([mne.EpochsArray(zero_block, missing_info, verbose=False)], force_update_info=True)
        epochs.info['bads'] = list(missing_targets)

    epochs.set_montage(MONTAGE_1005, match_case=False, on_missing='ignore')
    if missing_targets:
        epochs.interpolate_bads(reset_bads=True, mode='accurate', verbose=False)

    epochs.pick(TARGET_CHANNELS)
    epochs.reorder_channels(TARGET_CHANNELS)
    epochs.set_eeg_reference('average', projection=False, verbose=False)
    return epochs.get_data(copy=True).astype(np.float32), present_targets, missing_targets
