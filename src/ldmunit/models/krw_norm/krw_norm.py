#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 11:57:59 2019

@author: filipmelinscak
"""

import numpy as np
from scipy import stats
from sciunit.models import Model
from ...capabilities import ProducesLoglikelihood
import matlab.engine
import inspect
import os


class KrwNormModel(Model, ProducesLoglikelihood):

    def __init__(self, sigma_a, sigma_r, sigma_w, sigma_w0,
                 b0, b1, w0=None, name=None):
        if w0 is None:
            w0 = 0.0
        self.sigma_a = sigma_a
        self.sigma_r = sigma_r
        self.sigma_w = sigma_w
        self.sigma_w0 = sigma_w0
        self.b0 = b0
        self.b1 = b1
        self.w0 = w0
        super(KrwNormModel, self).__init__(name=name,
                                           sigma_a=sigma_a, sigma_r=sigma_r,
                                           sigma_w=sigma_w, sigma_w0=sigma_w0,
                                           b0=b0, b1=b1, w0=w0)

    def produce_loglikelihood(self, stimuli, rewards):
        class_path = os.path.dirname(inspect.getfile(type(self)))
        eng = matlab.engine.start_matlab()
        eng.addpath(class_path)
        stimuli = matlab.double(stimuli.tolist())
        rewards = matlab.double(rewards.tolist())
        # Calculate predictions in Matlab
        try:
            pred = eng.krw_norm_predict(stimuli, rewards,
                                        self.sigma_a, self.sigma_r,
                                        self.sigma_w, self.sigma_w0,
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
