function stats = model3(path_struct)
    bf.fhandle = @pspm_bf_psrf_fc;
    bf.args = [0, 0, 1];

    stats = fit_all(path_struct, bf, false, false);
end
