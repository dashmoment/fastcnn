#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 12:40:11 2017

@author: ubuntu
"""

import purgeinvalid_img as pi
import numpy as np

#pi.purgeinvalid_img("./ilsvrc11")
#
#pi.test();

a = np.array([[[3, 3, 3],[4, 4, 4]],[[1, 3, 3],[2, 4, 4]]])
b = np.array([[[8, 8, 3],[4, 4, 4]]])
#d = np.array([[6, 6, 6]])
c = np.append(a, b, axis=0)
#c = np.append(c, d, axis=0)