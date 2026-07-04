# ── Imports ──────────────────────────────────────────────────────────────────
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt

plt.rcParams.update({
    "font.size": 18,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 20,
    "figure.figsize": (10, 6),
    "figure.dpi": 200,
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
    "grid.alpha": 0.7
})

file_in = "your/folder"
apply_filter = True
filter_window = 5     # Median filter window size (must be odd)

# Load saved arrays
data = np.load(file_in)
t = data['t']
v = data['v']
R = data['R']
r_t = data['r_t']
k_t = data['k_t']
Rmax = data['Rmax']

# Extract velocity track
v_track = v[k_t]

# Apply median filter if requested
if apply_filter and filter_window > 0:
    w = filter_window if filter_window % 2 == 1 else filter_window + 1
    r_t = medfilt(r_t, kernel_size=w)
    v_track = medfilt(v_track, kernel_size=w)

ext = [t[0], t[-1], v[0], v[-1]]


# TFRgram
plt.figure(figsize=(12, 4))
im = plt.imshow(R, aspect="auto", origin="lower", extent=ext,
                cmap="inferno", vmin=0, vmax=Rmax)
# plt.plot(t, v_track, "k.", ms=1.5, alpha=0.5)
plt.xlabel(r"Time $t$ [s]")
plt.ylabel(r"Relative Velocity $v_r$ [m/s]")
plt.tight_layout()
cb = plt.colorbar(im)
cb.set_label(r"Range $R$ [m]")

# Extracted distance 
plt.figure()
plt.plot(t, r_t, 'b-')
plt.xlabel(r"Time $t$ [s]")
plt.ylabel(r"Range $R$ [m]")
plt.tight_layout()
plt.grid(True)

# Target velocity
plt.figure()
plt.plot(t, v_track, 'r-')
plt.xlabel(r"Time $t$ [s]")
plt.ylabel(r"Relative Velocity $v_r$ [m/s]")
plt.tight_layout()
plt.grid(True)

# Raw Doppler samples
xa = data['xa_list'][30].flatten()
xb = data['xb_list'][30].flatten()

# to illustraet mixer output in 5.5GHz radar
xa_xb = np.column_stack((xa, xb)).flatten()

Tprt = float(data['Tprt'])
t_samples = np.arange(len(xa)) * Tprt * 1000.0

plt.figure()
plt.plot(t_samples, (np.real(xa)-np.mean(xa)), marker='o', label="$\mathcal{Re(xa)}$")
# plt.plot(t_samples, np.imag(xa), marker='o', label="imag(xa)")
plt.plot(t_samples, (np.real(xb)-np.mean(xb)), marker='o', label="$\mathcal{Re(xb)}$")
# plt.plot(t_samples, np.imag(xb), marker='o', label="imag(xb)")

#test
# plt.plot(np.real(xa), np.real(xb), marker='o')

plt.xlabel(r"Time $t$ [ms]")
plt.ylabel(r"Amplitude")
plt.legend()
plt.tight_layout()
plt.grid(True)

#combine samples to illustrate mixer output before seperation
plt.figure()
# Interleaved samples happen roughly twice as fast over the frame
t_samples = np.arange(len(xa_xb)) * (Tprt / 2.0) * 1000.0
plt.plot(t_samples, (np.real(xa_xb) - np.mean(np.real(xa_xb))), marker='o', label="$\mathcal{Re(x)}$")

plt.xlabel(r"Time $t$ [ms]")
plt.ylabel(r"Amplitude")
plt.legend()
plt.tight_layout()
plt.grid(True)

plt.show()
