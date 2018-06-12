#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 21:19:16 2018

@author: wuyiming
"""


from librosa.core import load,stft,istft,magphase,note_to_hz,amplitude_to_db
from librosa.display import specshow
import numpy as np


SR = 44100
FFTSIZE = 4096
H =2048

def LoadAudio(path_audio):
    y, sr = load(path_audio, sr=SR)
    S_mag, _ = magphase(stft(y, n_fft=FFTSIZE, hop_length=H))
    
    return S_mag


def ReconstructAudio(S_mag, S_phase,norm):
    S = (S_mag * norm) * S_phase
    y = istft(S, hop_length=H, win_length=FFTSIZE)
    
    return y

def SegmentByBeat(pianoroll, bpm, beats):
    indices = np.round(np.arange(1, beats) * (60.0 / bpm) * SR / H).astype(np.int32)
    segments = np.split(pianoroll, indices, axis=1)
    return segments


def PlotSpec(S):
    fftsize = S.shape[0] * 2
    specshow(amplitude_to_db(S,ref=np.max), sr=SR * fftsize / FFTSIZE, y_axis="linear")
    
def PlotPianoroll(P, min_note="C1"):
    specshow(P, y_axis="cqt_note", fmin=note_to_hz(min_note))
    
def PlotTemplates(T):
    fftsize = T.shape[1] * 2
    specshow(amplitude_to_db(T.T), sr=SR * fftsize / FFTSIZE, y_axis="linear")