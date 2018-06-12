#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 18:44:18 2018

@author: wuyiming
"""

import util
import NMF
import chord
import NoteTemplate

import numpy as np

S = util.LoadAudio("PianoChord-70bpm.wav")
W = NoteTemplate.getNoteTemplates("audio_notes")

_,H = NMF.nmf_beta(S,48,W.T,beta=0.5,iteration=20)

util.PlotPianoroll(H)

H_binary = (H / np.max(H)) > 0.2
util.PlotPianoroll(H_binary)

segments = [np.sum(seg,axis=1) for seg in util.SegmentByBeat(H,70,4*4)]
chords = [chord.match_chord(seg/seg.max()) for seg in segments]

print(chords)