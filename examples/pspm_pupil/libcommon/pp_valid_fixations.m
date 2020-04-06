function sts = pp_valid_fixations(pupil_fpath, fixation_angle)
    for i = 1:numel(pupil_fpath)
        fpath = pupil_fpath{i};
        try
            [sts, outfile] = pspm_convert_pixel2unit(fpath, [3, 4, 6, 7], 'cm', 44.28, 24.91, 70.0);
        catch
            [sts, outfile] = pspm_convert_pixel2unit(fpath, [3, 4], 'cm', 44.28, 24.91, 70.0);
        end
        if sts ~= 1; return; end
        options.overwrite = true;
        [sts, outfile] = pspm_find_valid_fixations(fpath, fixation_angle, 70.0, 'cm', options);
        if sts ~= 1; return; end
    end
end
