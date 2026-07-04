import sys, time
import numpy as np
from scipy.constants import speed_of_light as c
from scipy.signal.windows import chebwin
import matplotlib.pyplot as plt

from ifxradarsdk import get_version_full
import ifxradarsdk.mimose as mimose
import ifxradarsdk.mimose.types as mtypes

"""
NOTE!: 
    Normally, a get_dac_offset function should be present to automatically set "dac" based on "Rmax".
"""

# Acquisition
T_acq = 20.0            # acquisition duration              [s]
P_thr = -45.0           # detection threshold               [dBFS]

# Radar timing
N     = 32              # slow-time samples per frame
Tprt  = 1.2e-3          # pulse repetition time              [s]
Tfr   = 0.08            # frame repetition time              [s]

# Signal processing
Nfft  = 128             # Doppler FFT size (zero-padded)
Rmax  = 5               # maximum unambiguous range          [m]
dt    = 130e-6          # inter-pulse delay PC0 -> PC1       [s]
R_off = 0.16            # static hardware range offset       [m]
dac   = 17              # VCO DAC offset for BFSK (MUST BE SET VIA THE MATLAB CODE bfsk_distance.m)

# Register map (hardcoded from bfsk_distance.m)
REG_DAC = [0x1010, 0x1011, 0x1012, 0x1013]   # VCO_PCx_DAC_OFFSET
REG_AGC = [0x3005, 0x3006, 0x3007, 0x3008]   # PCx_AGC

def doppler_fft(x, w, Nfft):
    xw = (x - np.mean(x)) * w
    return np.fft.fftshift(np.fft.fft(xw, n=Nfft)) / len(x)


def build_tfragram(Sa, Sb, f, Rmax, dt, R_off):
    """Phase-difference ranging at every (k, n) bin.
    """
    dphi = np.mod(np.angle(Sb) - np.angle(Sa), 2 * np.pi)
    R = (dphi / (2 * np.pi)) * Rmax

    # Dynamic compensation
    R += (-dt * f * Rmax)[:, np.newaxis] - R_off

    # Phase-wrap correction
    R[R < 0]    += Rmax
    R[R > Rmax] -= Rmax
    return R


def extract_target(S_dB, R, v, P_thr, Rmax, Nfft):
    """Peak-detection on the spectrogram → target range & velocity vs time."""
    Nt  = S_dB.shape[1]
    k0  = Nfft // 2                         # DC bin index

    r_t = np.zeros(Nt)
    v_t = np.zeros(Nt)
    k_t = np.full(Nt, k0, dtype=int)

    for n in range(Nt):
        k = int(np.argmax(S_dB[:, n]))

        if S_dB[k, n] > P_thr:
            k_t[n] = k
            r = R[k, n]
            # Reject wrapping artefacts near Rmax
            r_t[n] = r if r <= 0.833 * Rmax else 0.0
            v_t[n] = v[k]
        else:
            # No detection -> zero velocity, hold previous range
            r_t[n] = r_t[n - 1] if n > 0 else 0.0

    return r_t, v_t, k_t


print(f"radar sdk version: {get_version_full()}")
dev = mimose.DeviceMimose()

# Enable all 4 pulse configs
for i in range(4):
    dev.config.FrameConfig[0].selected_pulse_configs[i] = True

# TX/RX channels
ch1 = int(mtypes.ifx_Mimose_Channel_t.IFX_MIMOSE_CHANNEL_TX1_RX1)
ch2 = int(mtypes.ifx_Mimose_Channel_t.IFX_MIMOSE_CHANNEL_TX2_RX2)
dev.config.PulseConfig[0].channel = ch1
dev.config.PulseConfig[1].channel = ch1
dev.config.PulseConfig[2].channel = ch2
dev.config.PulseConfig[3].channel = ch2

# Max TX power, frame timing
for i in range(4):
    dev.config.PulseConfig[i].tx_power_level = 63
dev.config.FrameConfig[0].num_of_samples = N
dev.config.FrameConfig[0].pulse_repetition_time_s = Tprt
dev.config.FrameConfig[0].frame_repetition_time_s = Tfr
dev.set_config()

# Read back actual centre frequency
fc  = float(dev.config.AFC_Config.rf_center_frequency_Hz)
fs  = 1.0 / Tprt
lam = c / fc

print(f"f_c    = {fc/1e9} [GHz]")
print(f"N      = {N} Doppler samples")
print(f"T_PRT  = {Tprt*1e3} [ms]")
print(f"f_s    = {fs} [Hz]")
print(f"lambda = {lam*1e3} [mm]")
print(f"R_max  = {Rmax} [m]")

# BFSK offsets
dev.set_register(REG_DAC[0], dac)
dev.set_register(REG_DAC[1], 0)
dev.set_register(REG_DAC[2], dac)
dev.set_register(REG_DAC[3], 0)

# Enable AGC on all pulse configs
for reg in REG_AGC:
    dev.set_register(reg, 0x000F)

# Collect frames (antenna 1 only: rows 0 and 1)
xa_list = []     # freq A  (f + delta f)
xb_list = []     # freq B  (f)

print(f"\n Acquiring for {T_acq:.0f} s ...")
t0 = time.time()
nf = 0

while (time.time() - t0) < T_acq:
    
    data, meta = dev.get_next_frame()
    
    if data is None or data.size == 0:
        continue
    
    frame = data.squeeze()                   # (4, N) complex
    xa_list.append(frame[0, :].copy())       # Doppler samples for PC0
    xb_list.append(frame[1, :].copy())       # Doppler samples for PC1
    nf += 1
    
    sys.stdout.write(f"\r  {nf} frames / {time.time()-t0:.1f} s")
    sys.stdout.flush()

dev.stop_acquisition()

print(f"\n  Done! -> {nf} frames collected -> {nf*N} Doppler samples\n")

# Chebyshev window (60 dB sidelobe suppression)
w = chebwin(N, at=60)

# STFT matrices:  Sa[k, n] and Sb[k, n]
Sa = np.column_stack([doppler_fft(xa_list[n], w, Nfft) for n in range(nf)])
Sb = np.column_stack([doppler_fft(xb_list[n], w, Nfft) for n in range(nf)])

# Axes
t = np.arange(nf) * Tfr                                      # time       [s]
f = np.fft.fftshift(np.fft.fftfreq(Nfft, d=1.0 / fs))       # frequency  [Hz]
v = f * lam / 2                                               # velocity   [m/s]

# Spectrogram  (average power of both tones)
S = 10 * np.log10((np.abs(Sa)**2 + np.abs(Sb)**2) / 2 + 1e-30) # 1e-30 for no -inf log

# TFRgram  R[k, n]
R = build_tfragram(Sa, Sb, f, Rmax, dt, R_off)

# Target extraction
r_t, v_t, k_t = extract_target(S, R, v, P_thr, Rmax, Nfft)

out_filename = f"bfsk_data_{time.strftime('%Y%m%d_%H%M%S')}.npz"
print(f"Saving the raw data and results to {out_filename} ...")
np.savez(
    out_filename,
    # Raw Data
    xa_list=xa_list,
    xb_list=xb_list,
    # Axes
    t=t, f=f, v=v,
    # Processed Results
    Sa=Sa, Sb=Sb,
    S=S, R=R,
    r_t=r_t, v_t=v_t, k_t=k_t,
    # Metadata
    fs=fs, fc=fc, Tprt=Tprt, Rmax=Rmax
)
print("Save completed.\n")

ext = [t[0], t[-1], v[0], v[-1]]
v_track = v[k_t]

# 1 Spectrogram
fig1, ax1 = plt.subplots(figsize=(12, 4))

im0 = ax1.imshow(S, aspect="auto", 
                 origin="lower", 
                 extent=ext,
                 cmap="viridis",
                 vmin=np.percentile(S, 5), 
                 vmax=np.percentile(S, 99))

# ax1.plot(t, v_track, "r.", ms=1.5, alpha=0.5)
ax1.set_xlabel(r"$t$ [s]")
ax1.set_ylabel(r"$v_r$ [m/s]")
fig1.colorbar(im0, ax=ax1, label="A [dB]")
fig1.tight_layout()

# 2 TFRgram
fig2, ax2 = plt.subplots(figsize=(12, 4))

im1 = ax2.imshow(R, 
                 aspect="auto", 
                 origin="lower", 
                 extent=ext,
                 cmap="jet",
                 vmin=0, 
                 vmax=Rmax)

# ax2.plot(t, v_track, "k.", ms=1.5, alpha=0.5)
ax2.set_xlabel(r"$t$ [s]")
ax2.set_ylabel(r"$v_r$ [m/s]")
fig2.colorbar(im1, ax=ax2, label=r"$R$ [m]")
fig2.tight_layout()

# 3 Extracted distance vs time
fig3, ax3 = plt.subplots(figsize=(12, 4))
ax3.plot(t, r_t, "b-", lw=1)
ax3.set_xlabel(r"$t$ [s]")
ax3.set_ylabel(r"$R$ [m]")
ax3.set_ylim(0, Rmax)
ax3.grid()
fig3.tight_layout()

plt.show()
