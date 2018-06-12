#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 23:23:52 2018

@author: wuyiming
"""

import NMF
import util as U
import numpy as np
from librosa.util import find_files

def getNoteTemplates(path_notes):
    list_templates = []
    list_noteaudio = find_files(path_notes,ext="wav")
    for noteaudio in list_noteaudio:
        S_mag = U.LoadAudio(noteaudio)
        init_H = np.ones((1,S_mag.shape[1]))
        template,activate = NMF.nmf_sklearn(S_mag,k=1,H=init_H,verbose=False)
        list_templates.append(template[:,0]/np.max(template))
    templates = np.stack(list_templates)
    return templates


