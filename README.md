[![CI](https://github.com/hschn58/rPPG/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/hschn58/rPPG/actions/workflows/ci.yml)
[![Last Commit](https://img.shields.io/github/last-commit/hschn58/rPPG)](https://github.com/hschn58/rPPG/commits/main)

![Python](https://img.shields.io/badge/python-3.10%20|%203.11-blue)
![Platform](https://img.shields.io/badge/platform-linux%20%7C%20macOS-lightgrey)
![License](https://img.shields.io/badge/license-MIT-informational)

# rPPG — Remote Photoplethysmography (Face Video → Heart Rate)

This subproject extracts heart-rate signals from face video with a focus on **robustness to motion and lighting**.  
The pipeline has evolved from basic spectral estimation toward a more complete stack that integrates:

- **Edge-model AI for ROI detection**: lightweight on-device models provide robust face/ROI timestamping.  
- **Forward–backward Lucas–Kanade optical flow** to stabilize facial ROIs across frames.  
- **Standard-deviation motion metrics** to down-weight or reject unstable segments.  
- **Aperture, exposure, and white-balance consistency checks** to minimize camera-induced artifacts.  
- **Spectral estimators** (FFT peak, wavelet ridge, Butterworth band-pass) applied on the stabilized, pre-filtered signal.  

---

## Features

- **ROI stabilization** with forward–backward Lucas–Kanade optical flow  
- **Motion-aware selection**: standard deviation of flow/intensity used as a rejection/weighting criterion  
- **Camera consistency handling**: aperture/exposure/WB checks preserve physiological variation over environmental noise  
- **Multiple estimators**: FFT, wavelet, and Butterworth filtering for heart-rate extraction  
- **Outputs**: BPM time series with optional confidence metric, plots for each method, and debug visuals showing ROI tracking  


---

## Full_Stack Pipeline

The [`Full_Stack/`](./Full_Stack) directory contains the **culmination of the rPPG pipeline** — combining all major components into a single, reproducible workflow.  

### Key additions
- **End-to-end integration**: from video input → ROI stabilization → motion filtering → heart-rate estimation.  
- **Unified interface**: a single driver script to run preprocessing, optical flow stabilization, motion rejection, and spectral analysis together.  
- **Configurable parameters**: window size, overlap, frequency band, detector type, and motion thresholds adjustable in one place.  
- **Output suite**: synchronized CSV logs, BPM plots, and debug visuals (ROI trajectories, motion metrics).  
- **Scalability**: designed for batch processing of multiple clips with consistent settings.  

This folder demonstrates how the individual methods explored earlier (optical flow, motion metrics, spectral estimators, and camera consistency checks) come together in a **complete rPPG pipeline**.

## Multi-Region Heart Rate Estimation (CHROM & POS)

As an extension of the core pipeline, we evaluated multiple face regions simultaneously to compare algorithm performance under real-world conditions. The test used five distinct facial ROIs extracted from a video clip, with heart rate (BPM) and signal-to-noise ratio (SNR) estimated via:

CHROM (with and without motion processing)

POS (no motion processing)

Each region’s signal quality (raw intensity) and reconstructed waveform were analyzed across ~300 frames, estimating both heart rate (BPM) and signal‑to‑noise ratio (SNR). In scenarios with stable lighting and clear facial texture, POS without motion correction achieved higher precision than CHROM methods. 

Notably, when motion processing was applied in CHROM, a recurring 2.5 Hz artifact (~150 BPM) appeared in multiple regions. The signal is high-fidelity and stable across time, yet does not correspond to any plausible physiological source. The root cause is currently unknown — potential factors include cumulative error in optical flow stabilization, bounding box jitter, or framewise interpolation drift.

While this frequency is characteristic of certain pathological tremors (e.g. Holmes tremor), no data acquisition volunteers are known to exhibit such conditions.

![Complete pipeline driver output](./Full_Result_Example.pdf)


## License

This project is licensed under the [MIT License](./LICENSE).

