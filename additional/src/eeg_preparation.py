"""EDF support loading, quality control, and linear band-power reduction."""
from pathlib import Path
import numpy as np
from preregistered_analysis import REQUIRED_CHANNELS, NumericalFailure
ABS_UV=500.0
P2P_UV=1000.0
STEP_UV=200.0
FLAT_SD_UV=0.1
IDENTICAL_RUN_MS=100.0

def _longest_equal_run(samples: np.ndarray) -> int:
    if samples.size < 2:
        return int(samples.size)
    changes = np.flatnonzero(np.diff(samples) != 0)
    if changes.size == 0:
        return int(samples.size)
    boundaries = np.concatenate(([-1], changes, [samples.size - 1]))
    return int(np.diff(boundaries).max())


def time_domain_qc_mask(required_support_uvolts: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Apply the registered time-domain QC rules to the 42-cell support."""
    data = np.asarray(required_support_uvolts, dtype=float)
    samples_per_cell = int(round(4.0 * float(sampling_rate)))
    if data.shape != (5, 42 * samples_per_cell):
        raise ValueError("QC input must be five channels over the exact 168-second support")
    mastoid_mean = (data[3] + data[4]) / 2.0
    posterior = data[:3] - mastoid_mean
    identical_run_samples = int(round(IDENTICAL_RUN_MS * sampling_rate / 1000.0))
    mask = np.ones(42, dtype=bool)
    for cell in range(42):
        start = cell * samples_per_cell
        stop = start + samples_per_cell
        raw_window = data[:, start:stop]
        posterior_window = posterior[:, start:stop]
        centered = posterior_window - np.median(posterior_window, axis=1, keepdims=True)
        failures = (
            not np.isfinite(raw_window).all()
            or np.nanmax(np.abs(centered)) > ABS_UV
            or np.nanmax(np.ptp(posterior_window, axis=1)) > P2P_UV
            or np.nanmax(np.abs(np.diff(posterior_window, axis=1))) > STEP_UV
            or np.nanmin(np.std(raw_window, axis=1)) < FLAT_SD_UV
            or any(_longest_equal_run(channel) >= identical_run_samples for channel in raw_window)
        )
        mask[cell] = not failures
    return mask


def _load_required_support(raw_path: Path, reader: str) -> tuple[np.ndarray, float]:
    """Load only [8,176) s from five required channels; no filtering/resampling."""
    import mne

    if reader == "edf":
        raw = mne.io.read_raw_edf(raw_path, preload=False, verbose="ERROR")
    elif reader == "brainvision":
        raw = mne.io.read_raw_brainvision(raw_path, preload=False, verbose="ERROR")
    else:
        raise ValueError(reader)
    missing = [channel for channel in REQUIRED_CHANNELS if channel not in raw.ch_names]
    if missing:
        raise NumericalFailure(f"missing required channels: {','.join(missing)}")
    sampling_rate = float(raw.info["sfreq"])
    start = int(round(8.0 * sampling_rate))
    stop = int(round(176.0 * sampling_rate))
    if raw.n_times < int(round(180.0 * sampling_rate)):
        raise NumericalFailure("recording is shorter than the required 180 seconds")
    support_volts = raw.get_data(picks=list(REQUIRED_CHANNELS), start=start, stop=stop)
    expected = (5, int(round(168.0 * sampling_rate)))
    if support_volts.shape != expected:
        raise NumericalFailure(f"required support shape {support_volts.shape} != {expected}")
    if not np.isfinite(support_volts).all():
        raise NumericalFailure("nonfinite sample in required support")
    return support_volts, sampling_rate


def cell_band_powers(psd, frequency):
    """Return cell x (alpha,total); linear reduction preserves PSD aggregation."""
    psd=np.asarray(psd,dtype=float)
    frequency=np.asarray(frequency)
    if psd.ndim!=3 or psd.shape[:2]!=(3,42) or psd.shape[-1]!=len(frequency):
        raise ValueError('Expected 3 x 42 x frequency PSD array')
    roi=psd.mean(axis=0)
    alpha=(frequency>=8)&(frequency<=13)
    total=(frequency>=1)&(frequency<=30)
    return np.stack([np.trapezoid(roi[:,alpha],frequency[alpha],axis=-1),np.trapezoid(roi[:,total],frequency[total],axis=-1)],axis=-1)
