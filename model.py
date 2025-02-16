import librosa
import numpy as np
import soundfile as sf
import os
import csv
import argparse

def interpolate_time(idxs: np.ndarray, arr):
    start = np.floor(idxs).astype(int) 
    frac = (idxs - start)[None, None, :]
    
    
    start = np.clip(start, 0, arr.shape[2] - 2)
    
    shifted_arr = np.concatenate((arr[:, :, 1:], np.zeros((arr.shape[0], arr.shape[1], 1))), axis=2)
    return arr[:, :, start] * (1 - frac) + shifted_arr[:, :, start] * frac

def round_interpolate_time(idxs: np.ndarray, arr):
    idxs = np.clip((idxs + 0.5).astype(int), 0, arr.shape[2] - 1)  # Prevent out-of-bounds
    return arr[:, :, idxs]

pitch_shift = -5
n_fft = 1024
hop_len = 256
win_len = 1024

def anonymize(input_audio_path): 
    waveform, sr = librosa.load(input_audio_path, sr=None, mono=False)
    original_length = waveform.shape[1] if waveform.ndim > 1 else waveform.shape[0]

    if waveform.ndim == 1:
        waveform = np.expand_dims(waveform, axis=0)  
    
    anls_stft = librosa.stft(waveform, n_fft=n_fft, hop_length=hop_len, win_length=win_len)
    channels, n_anls_freqs, n_anls_frames = anls_stft.shape
    
    scaling = 2 ** (pitch_shift / 12)
    n_synth_frames = np.floor(n_anls_frames * scaling).astype(int)
    synth_frames = np.arange(n_synth_frames)
    og_idxs = np.minimum(synth_frames / scaling, n_anls_frames - 1)
    
    mags = np.abs(anls_stft)
    phases = np.angle(anls_stft)
    phase_diffs = np.mod(phases - np.concatenate((np.zeros((channels, n_anls_freqs, 1)), phases[:, :, :-1]), axis=2), np.pi * 2)
    
    shifted_mags = interpolate_time(og_idxs, mags)
    shifted_phase_diffs = interpolate_time(og_idxs, phase_diffs)
    unshifted_phases = round_interpolate_time(og_idxs, phases)
    
    shifted_phases = np.zeros((channels, n_anls_freqs, n_synth_frames))
    shifted_phases[:, :, 0] = shifted_phase_diffs[:, :, 0]
    epsilon = 1e-8  
    for t in range(1, n_synth_frames):
        time_phases = shifted_phases[:, :, t - 1] + shifted_phase_diffs[:, :, t]
        freq_phases = unshifted_phases[:, :, t]
        transient = (shifted_mags[:, :, t] - shifted_mags[:, :, t - 1]) / (np.maximum(shifted_mags[:, :, t] + shifted_mags[:, :, t - 1], epsilon))
        transient[transient < 0.5] = 0
        transient[transient >= 0.5] = 1
        shifted_phases[:, :, t] = np.mod(freq_phases * transient + time_phases * (1 - transient), np.pi * 2)
    
    synth_stft = shifted_mags * np.exp(shifted_phases * 1j)
    new_waveform = librosa.istft(synth_stft, hop_length=hop_len, win_length=win_len, n_fft=n_fft)

    
    new_waveform = librosa.resample(new_waveform, orig_sr=sr, target_sr=sr * (len(new_waveform.T) / original_length))
    audio=new_waveform.T
    sr=sr
    return audio, sr

