function sts = pp_pfe(pupil_fpath)
    opt.mode = 'auto';
    opt.C_z = 525;
    for i = 1:numel(pupil_fpath)
        fpath = pupil_fpath{i};
        try
            [sts, ~] = pspm_convert_pixel2unit(fpath, [3, 4, 6, 7], 'mm', 442.8, 249.1, 700);
        catch
            [sts, ~] = pspm_convert_pixel2unit(fpath, [3, 4], 'mm', 442.8, 249.1, 700);
        end
        if sts ~= 1; return; end
        opt.overwrite = true;
        sts = pspm_pupil_correct_eyelink(fpath, opt);
        if sts ~= 1; return; end;
    end
end
