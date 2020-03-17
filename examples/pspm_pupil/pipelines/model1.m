function stats = model1(path_struct)
    bf.fhandle = @pspm_bf_psrf_fc;
    bf.args = [1, 1, 0];

    stats = fit_all(path_struct, bf, false, false);
end
