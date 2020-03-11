function out = model4(filepath)
    bf.fhandle = @pspm_bf_psrf_fc;
    bf.args = [1, 0, 1];

    out = fit(filepath, bf, true, false);
    out = 0;
end
