#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 14:00:43 2018

@author: wuyiming
"""

import numpy as np
from librosa.filters import cq_to_chroma
from scipy.spatial.distance import euclidean, correlation

"""
Prepare chord templates
"""

temp_C = np.array([[1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
                   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
                   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                   [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                   [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
                   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
                   [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1]])

qualities = ["", "min", "dim", "dim", "aug", "7", "min7", "maj7"]

notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

mat_templates = np.zeros((12*len(qualities), 12))

chromafilter = cq_to_chroma(88)

for q in range(len(qualities)):
    for r in range(12):
        mat_templates[12*q+r, :] = np.roll(temp_C[q, :], r)


def match_chord(segment, thld=0.2):
    """
    Compare the segment note information with chord templates
    """
    bassnote = np.min(np.nonzero(segment > (segment.max()*thld))[0])
    bassnote_chroma = bassnote % 12
    n_notes = segment.size
    chroma = np.dot(chromafilter[:, :n_notes], segment[:, None])
    chroma /= chroma.max()
    dist = np.array([correlation(mat_templates[i, :], chroma) for i in range(mat_templates.shape[0])])
    id_chord = np.argmin(dist)
    root = id_chord % 12
    qual = id_chord // 12
    chordtext = notes[root] + ":" + qualities[qual]
    if bassnote_chroma != root:
        chordtext = chordtext + "/" + notes[bassnote_chroma]
    return chordtext
