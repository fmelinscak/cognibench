function stats = model4(path_struct)
    bf.fhandle = @pspm_bf_psrf_fc;
    bf.args = [1, 0, 1];

    stats = fit_all(path_struct, bf, true, false);
end
