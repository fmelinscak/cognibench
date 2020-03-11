function out = model5(filepath)
    bf.fhandle = @pspm_bf_psrf_fc;
    bf.args = [1, 0, 1];

    out = fit(filepath, bf, true, true);
    out = 0;
end
