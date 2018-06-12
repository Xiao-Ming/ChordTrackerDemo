#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 13:17:35 2018

@author: wuyiming
"""

import numpy as np
from librosa.core import load
from librosa.output import write_wav
from scipy.io.wavfile import write,read


SR=44100

def loadsplit(path_split):
    f = open(path_split)
    line = f.readline()
    list_split = []
    while line != "":
        list_split.append(float(line.split()[0]) * SR)
        line = f.readline()
        
    return np.round(list_split).astype(np.int32)


sr, y = read("single_notes.WAV")
split = loadsplit("silence.txt")

y_list = np.split(y, split)

for note in range(len(y_list)):
    fname = "audio_notes/note_%02d.wav" % note
    write(fname,SR,y_list[note])