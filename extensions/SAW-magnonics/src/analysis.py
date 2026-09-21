"""Analysis tools for magnon-phonon coupling simulations.

Provides:
- Local spectral analysis (windowed FFT)
- Linewidth extraction from ringdown / frequency-domain fits
- Complex eigenfrequency estimation
- Branch tracking and anticrossing detection
- Phase-lag mapping
- Exceptional point identification
"""

import numpy as np
from scipy.signal import welch, windows
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Spectral analysis
# ---------------------------------------------------------------------------

def local_fft(signal_2d, dt, dx, window='hann'):
    """Compute 2D FFT of a spatiotemporal signal.

    Parameters
    ----------
    signal_2d : ndarray, shape (Nt, Nx)
        Time-space signal (e.g., m_y(t, x)).
    dt : float
        Time step (s).
    dx : float
        Spatial step (m).
    window : str
        Window function for temporal axis.

    Returns
    -------
    freqs : ndarray
        Frequency axis (Hz), positive only.
    kvecs : ndarray
        Wavenumber axis (rad/m), full.
    power : ndarray, shape (Nf, Nk)
        Power spectral density.
    """
    Nt, Nx = signal_2d.shape

    # Apply temporal window
    win = windows.get_window(window, Nt)
    windowed = signal_2d * win[:, np.newaxis]

    # 2D FFT
    ft = np.fft.fft2(windowed)
    ft = np.fft.fftshift(ft)
    power = np.abs(ft)**2

    freqs = np.fft.fftshift(np.fft.fftfreq(Nt, dt))
    kvecs = np.fft.fftshift(np.fft.fftfreq(Nx, dx)) * 2 * np.pi

    return freqs, kvecs, power


def windowed_local_spectrum(signal_2d, dt, x_positions, dx,
                            window_width_cells=32, overlap=0.5):
    """Compute local spectra at different spatial positions using windowed FFT.

    Slides a spatial window along x, computes FFT within each window,
    and assigns the result to the window center position.

    Parameters
    ----------
    signal_2d : ndarray, shape (Nt, Nx)
        Time-space signal.
    dt : float
        Time step (s).
    x_positions : ndarray
        Full array of x positions (m).
    dx : float
        Spatial cell size (m).
    window_width_cells : int
        Width of spatial window in cells.
    overlap : float
        Fractional overlap between windows (0 to 1).

    Returns
    -------
    x_centers : ndarray
        Center positions of windows (m).
    freqs : ndarray
        Frequency axis (Hz).
    local_spectra : ndarray, shape (N_windows, Nf)
        Power spectrum at each position.
    """
    Nt, Nx = signal_2d.shape
    step = max(1, int(window_width_cells * (1 - overlap)))

    x_centers = []
    local_spectra = []

    # Temporal window
    t_win = windows.hann(Nt)

    for i_start in range(0, Nx - window_width_cells + 1, step):
        i_end = i_start + window_width_cells
        x_center = x_positions[i_start + window_width_cells // 2]

        # Extract spatial window
        chunk = signal_2d[:, i_start:i_end]

        # Apply spatial and temporal windows
        s_win = windows.hann(window_width_cells)
        windowed = chunk * t_win[:, np.newaxis] * s_win[np.newaxis, :]

        # Sum over spatial cells (get local temporal spectrum)
        local_signal = np.sum(windowed, axis=1)

        # FFT
        spectrum = np.abs(np.fft.rfft(local_signal))**2
        x_centers.append(x_center)
        local_spectra.append(spectrum)

    freqs = np.fft.rfftfreq(Nt, dt)
    return np.array(x_centers), freqs, np.array(local_spectra)


# ---------------------------------------------------------------------------
# Linewidth and frequency extraction
# ---------------------------------------------------------------------------

def lorentzian(f, f0, gamma, A, C):
    """Lorentzian spectral line shape.

    Parameters
    ----------
    f : array
        Frequency (Hz).
    f0 : float
        Center frequency (Hz).
    gamma : float
        Half-width at half-maximum (Hz).
    A : float
        Peak amplitude.
    C : float
        Background offset.
    """
    return A * gamma**2 / ((f - f0)**2 + gamma**2) + C


def fit_lorentzian(freqs, spectrum, f_range=None, p0=None):
    """Fit a Lorentzian to a spectral peak.

    Parameters
    ----------
    freqs : array
        Frequency axis (Hz).
    spectrum : array
        Power spectrum.
    f_range : tuple (f_min, f_max) or None
        Frequency range for fitting.
    p0 : tuple or None
        Initial guess (f0, gamma, A, C).

    Returns
    -------
    dict with keys:
        'f0': center frequency (Hz)
        'gamma': half-linewidth (Hz)
        'fwhm': full-width at half-maximum (Hz)
        'amplitude': peak amplitude
        'success': bool
        'popt': fitted parameters
        'pcov': covariance matrix
    """
    if f_range is not None:
        mask = (freqs >= f_range[0]) & (freqs <= f_range[1])
        f_fit = freqs[mask]
        s_fit = spectrum[mask]
    else:
        f_fit = freqs
        s_fit = spectrum

    if p0 is None:
        i_peak = np.argmax(s_fit)
        f0_guess = f_fit[i_peak]
        A_guess = np.max(s_fit) - np.min(s_fit)
        gamma_guess = (f_fit[-1] - f_fit[0]) / 20
        C_guess = np.min(s_fit)
        p0 = (f0_guess, gamma_guess, A_guess, C_guess)

    try:
        popt, pcov = curve_fit(lorentzian, f_fit, s_fit, p0=p0,
                               maxfev=5000,
                               bounds=([f_fit[0], 0, 0, 0],
                                       [f_fit[-1], np.inf, np.inf, np.inf]))
        return {
            'f0': popt[0],
            'gamma': abs(popt[1]),
            'fwhm': 2 * abs(popt[1]),
            'amplitude': popt[2],
            'background': popt[3],
            'success': True,
            'popt': popt,
            'pcov': pcov,
        }
    except (RuntimeError, ValueError):
        return {
            'f0': p0[0],
            'gamma': p0[1],
            'fwhm': 2 * p0[1],
            'amplitude': p0[2],
            'background': p0[3] if len(p0) > 3 else 0,
            'success': False,
            'popt': p0,
            'pcov': None,
        }


def extract_peaks(freqs, spectrum, n_peaks=2, prominence=0.1):
    """Find and fit multiple peaks in a spectrum.

    Parameters
    ----------
    freqs : array
        Frequency axis (Hz).
    spectrum : array
        Power spectrum.
    n_peaks : int
        Maximum number of peaks to find.
    prominence : float
        Minimum prominence relative to max.

    Returns
    -------
    list of dict
        Fitted peak parameters, sorted by frequency.
    """
    from scipy.signal import find_peaks as sp_find_peaks

    # Normalize
    s_norm = spectrum / np.max(spectrum)
    peaks, properties = sp_find_peaks(s_norm, prominence=prominence,
                                       distance=5)

    if len(peaks) == 0:
        return []

    # Sort by prominence and take top n_peaks
    prom = properties['prominences']
    idx_sorted = np.argsort(prom)[::-1][:n_peaks]
    peak_indices = sorted(peaks[idx_sorted])

    results = []
    for i_peak in peak_indices:
        f_center = freqs[i_peak]
        # Define fitting range around peak
        df = (freqs[-1] - freqs[0]) / 10
        f_range = (max(freqs[0], f_center - df),
                   min(freqs[-1], f_center + df))
        fit = fit_lorentzian(freqs, spectrum, f_range=f_range)
        results.append(fit)

    return sorted(results, key=lambda r: r['f0'])


# ---------------------------------------------------------------------------
# Complex eigenfrequency and EP detection
# ---------------------------------------------------------------------------

def complex_eigenfrequencies(freqs, spectrum, n_modes=2, f_range=None):
    """Extract complex eigenfrequencies from spectral data.

    The real part is the oscillation frequency, the imaginary part is
    the damping rate (linewidth / 2).

    omega_complex = omega_re - i * gamma

    Parameters
    ----------
    freqs : array
        Frequency axis (Hz).
    spectrum : array
        Power spectrum.
    n_modes : int
        Number of modes to extract.
    f_range : tuple or None
        Frequency range to search.

    Returns
    -------
    list of complex
        Complex eigenfrequencies (Hz).
    """
    if f_range is not None:
        mask = (freqs >= f_range[0]) & (freqs <= f_range[1])
        f_sub = freqs[mask]
        s_sub = spectrum[mask]
    else:
        f_sub = freqs
        s_sub = spectrum

    peaks = extract_peaks(f_sub, s_sub, n_peaks=n_modes)

    eigenfreqs = []
    for p in peaks:
        if p['success']:
            omega = 2 * np.pi * p['f0'] - 1j * 2 * np.pi * p['gamma']
        else:
            omega = 2 * np.pi * p['f0']
        eigenfreqs.append(omega)

    return eigenfreqs


def ep_proximity_metric(omega1, omega2):
    """Compute exceptional point proximity metric.

    At an EP, two complex eigenfrequencies coalesce: omega1 = omega2.
    The EP proximity is defined as:

        d_EP = |omega1 - omega2| / max(|Im(omega1)|, |Im(omega2)|)

    Values:
        d_EP >> 1: far from EP (well-separated modes)
        d_EP ~ 1: near EP
        d_EP << 1: at or past EP (modes coalesced)

    Parameters
    ----------
    omega1, omega2 : complex
        Complex eigenfrequencies.

    Returns
    -------
    float
        EP proximity metric.
    """
    delta = abs(omega1 - omega2)
    linewidth_scale = max(abs(omega1.imag), abs(omega2.imag), 1e-20)
    return delta / linewidth_scale


def classify_coupling_regime(freq_splitting, linewidth_avg):
    """Classify magnon-phonon coupling as weak, EP, or strong.

    Parameters
    ----------
    freq_splitting : float
        Frequency splitting between two modes (Hz).
    linewidth_avg : float
        Average linewidth of the two modes (Hz).

    Returns
    -------
    str
        'strong', 'EP', or 'weak'.
    float
        Cooperativity C = (2*g)^2 / (kappa1 * kappa2).
    """
    if linewidth_avg <= 0:
        return 'undetermined', 0.0

    ratio = freq_splitting / linewidth_avg

    if ratio > 1.0:
        regime = 'strong'
    elif ratio > 0.3:
        regime = 'EP-proximal'
    else:
        regime = 'weak'

    cooperativity = ratio**2

    return regime, cooperativity


# ---------------------------------------------------------------------------
# Branch tracking
# ---------------------------------------------------------------------------

def track_branches(param_values, spectra_list, freqs, n_branches=2,
                   f_range=None):
    """Track spectral branches across a parameter sweep.

    Parameters
    ----------
    param_values : array
        Parameter being swept (e.g., B_ext or k_SAW).
    spectra_list : list of arrays
        Power spectra at each parameter value.
    freqs : array
        Frequency axis (Hz).
    n_branches : int
        Number of branches to track.
    f_range : tuple or None
        Frequency range for peak search.

    Returns
    -------
    dict with keys:
        'param': parameter values
        'branches': list of arrays, each (N_param,) with NaN for missing
        'linewidths': same structure for linewidths
        'regimes': coupling regime at each parameter value
    """
    branches = [np.full(len(param_values), np.nan) for _ in range(n_branches)]
    linewidths = [np.full(len(param_values), np.nan) for _ in range(n_branches)]
    regimes = []

    for i, spectrum in enumerate(spectra_list):
        peaks = extract_peaks(freqs, spectrum, n_peaks=n_branches)

        for j, peak in enumerate(peaks):
            if j < n_branches and peak['success']:
                branches[j][i] = peak['f0']
                linewidths[j][i] = peak['fwhm']

        # Classify coupling regime
        if len(peaks) >= 2 and peaks[0]['success'] and peaks[1]['success']:
            splitting = abs(peaks[1]['f0'] - peaks[0]['f0'])
            avg_lw = (peaks[0]['fwhm'] + peaks[1]['fwhm']) / 2
            regime, C = classify_coupling_regime(splitting, avg_lw)
        else:
            regime, C = 'single-peak', 0.0
        regimes.append((regime, C))

    return {
        'param': param_values,
        'branches': branches,
        'linewidths': linewidths,
        'regimes': regimes,
    }


# ---------------------------------------------------------------------------
# Ringdown analysis
# ---------------------------------------------------------------------------

def ringdown_fit(time, signal, f_guess=None):
    """Fit a damped oscillation to extract frequency and decay rate.

    Model: A * exp(-gamma * t) * cos(2*pi*f*t + phi) + C

    Parameters
    ----------
    time : array
        Time array (s).
    signal : array
        Oscillation signal (e.g., m_y(t)).
    f_guess : float or None
        Initial frequency guess (Hz). If None, estimated from FFT.

    Returns
    -------
    dict
        'frequency': oscillation frequency (Hz)
        'decay_rate': exponential decay rate (1/s)
        'linewidth': FWHM linewidth (Hz) = decay_rate / pi
        'amplitude': initial amplitude
        'success': bool
    """
    # Estimate frequency from FFT if not provided
    if f_guess is None:
        ft = np.abs(np.fft.rfft(signal))
        freqs = np.fft.rfftfreq(len(signal), time[1] - time[0])
        i_peak = np.argmax(ft[1:]) + 1
        f_guess = freqs[i_peak]

    def model(t, A, gamma, f, phi, C):
        return A * np.exp(-gamma * t) * np.cos(2 * np.pi * f * t + phi) + C

    try:
        p0 = [np.max(np.abs(signal)), 1e8, f_guess, 0, np.mean(signal)]
        popt, pcov = curve_fit(model, time, signal, p0=p0, maxfev=10000)
        return {
            'frequency': abs(popt[2]),
            'decay_rate': abs(popt[1]),
            'linewidth': abs(popt[1]) / np.pi,
            'amplitude': abs(popt[0]),
            'phase': popt[3],
            'offset': popt[4],
            'success': True,
        }
    except (RuntimeError, ValueError):
        return {
            'frequency': f_guess,
            'decay_rate': 0,
            'linewidth': 0,
            'amplitude': 0,
            'phase': 0,
            'offset': 0,
            'success': False,
        }


# ---------------------------------------------------------------------------
# Spatial maps
# ---------------------------------------------------------------------------

def compute_spatial_frequency_map(signal_2d, dt, x_positions,
                                   window_cells=32, f_range=None):
    """Compute position-resolved frequency and linewidth maps.

    Parameters
    ----------
    signal_2d : ndarray (Nt, Nx)
        Spatiotemporal signal.
    dt : float
        Time step (s).
    x_positions : ndarray (Nx,)
        Spatial positions (m).
    window_cells : int
        Spatial window width for local FFT.
    f_range : tuple or None
        Frequency range for peak search (Hz).

    Returns
    -------
    dict with spatial maps of frequency, linewidth, amplitude, regime.
    """
    dx = x_positions[1] - x_positions[0] if len(x_positions) > 1 else 1.0

    x_centers, freqs, spectra = windowed_local_spectrum(
        signal_2d, dt, x_positions, dx,
        window_width_cells=window_cells, overlap=0.5)

    n_pos = len(x_centers)
    freq_map = np.full((n_pos, 2), np.nan)
    lw_map = np.full((n_pos, 2), np.nan)
    amp_map = np.full((n_pos, 2), np.nan)
    regime_map = []

    for i in range(n_pos):
        peaks = extract_peaks(freqs, spectra[i], n_peaks=2)

        for j, p in enumerate(peaks[:2]):
            if p['success']:
                freq_map[i, j] = p['f0']
                lw_map[i, j] = p['fwhm']
                amp_map[i, j] = p['amplitude']

        if len(peaks) >= 2 and peaks[0]['success'] and peaks[1]['success']:
            splitting = abs(peaks[1]['f0'] - peaks[0]['f0'])
            avg_lw = (peaks[0]['fwhm'] + peaks[1]['fwhm']) / 2
            regime, C = classify_coupling_regime(splitting, avg_lw)
        else:
            regime, C = 'single', 0.0
        regime_map.append((regime, C))

    return {
        'x_centers': x_centers,
        'freqs': freqs,
        'spectra': spectra,
        'freq_map': freq_map,
        'linewidth_map': lw_map,
        'amplitude_map': amp_map,
        'regime_map': regime_map,
    }
