#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:29:33 2019

@author: filipmelinscak
"""

import numpy as np
from scipy import stats
from sciunit.models import Model
from ...capabilities import ProducesLoglikelihood
import matlab.engine
import inspect
import os


class RwNormModel(Model, ProducesLoglikelihood):

    def __init__(self, alpha, sigma, b0, b1, w0=None, name=None):
        if w0 is None:
            w0 = 0.0
        self.alpha = alpha
        self.sigma = sigma
        self.b0 = b0
        self.b1 = b1
        self.w0 = w0
        super(RwNormModel, self).__init__(name=name,
                                          alpha=alpha, sigma=sigma,
                                          b0=b0, b1=b1, w0=w0)

    def produce_loglikelihood(self, stimuli, rewards):
        class_path = os.path.dirname(inspect.getfile(type(self)))
        eng = matlab.engine.start_matlab()
        eng.addpath(class_path)
        stimuli = matlab.double(stimuli.tolist())
        rewards = matlab.double(rewards.tolist())
        # Calculate predictions in Matlab
        try:
            pred = eng.rw_norm_predict(stimuli, rewards,
                                       self.alpha, self.sigma,
                                       self.b0, self.b1, self.w0, nargout=2)
        finally:
            eng.exit()
        # Unpack predictions
        mu_pred_mat, sd_pred_mat = pred
        mu_pred = np.array(mu_pred_mat._data.tolist())  # TODO: find better way
        mu_pred = mu_pred.reshape(mu_pred_mat.size).transpose()
        sd_pred = np.array(sd_pred_mat._data.tolist())
        sd_pred = sd_pred.reshape(sd_pred_mat.size).transpose()

        # Create logpdf
        def logpdf(actions):
            pointwise_logpdf = stats.norm.logpdf(actions,
                                                 loc=mu_pred, scale=sd_pred)
            return np.sum(pointwise_logpdf)

        return logpdf
