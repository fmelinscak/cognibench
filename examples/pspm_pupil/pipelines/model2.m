function out = model2(filepath)
    bf.fhandle = @pspm_bf_psrf_fc;
    bf.args = [1, 0, 1];

    out = fit(filepath, bf, false, false);
    out = 0;
end
