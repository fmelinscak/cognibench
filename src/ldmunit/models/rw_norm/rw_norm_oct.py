#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:18:35 2019

@author: filipmelinscak
"""

import numpy as np
from scipy import stats
from sciunit.models import Model
from ...capabilities import ProducesLoglikelihood
import oct2py
import inspect
import os


class RwNormOctModel(Model, ProducesLoglikelihood):

    def __init__(self, alpha, sigma, b0, b1, w0=None, name=None):
        if w0 is None:
            w0 = 0.0
        self.alpha = alpha
        self.sigma = sigma
        self.b0 = b0
        self.b1 = b1
        self.w0 = w0
        super(RwNormOctModel, self).__init__(name=name,
                                             alpha=alpha, sigma=sigma,
                                             b0=b0, b1=b1, w0=w0)

    def produce_loglikelihood(self, stimuli, rewards):
        class_path = os.path.dirname(inspect.getfile(type(self)))

        # Calculate predictions in Octave
        with oct2py.Oct2Py() as oc:
            oc.addpath(class_path)
            mu_pred, sd_pred = oc.rw_norm_predict(stimuli, rewards,
                                                  self.alpha, self.sigma,
                                                  self.b0, self.b1, self.w0,
                                                  nout=2)

        # Create logpdf
        def logpdf(actions):
            pointwise_logpdf = stats.norm.logpdf(actions,
                                                 loc=mu_pred, scale=sd_pred)
            return np.sum(pointwise_logpdf)

        return logpdf
