function out = model0(filepath)
    bf.fhandle = @pspm_bf_psrf_fc;
    bf.args = [1, 0, 0];

    out = fit(filepath, bf, false, false);
    out = 0;
end
