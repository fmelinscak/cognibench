function sts = pp_pfe(pupil_fpath)
    % Perform pupil foreshortening error correction on the data of a subject.
    % If the gaze channels are in pixels, they are first converted to milimeters.
    %
    % Parameters
    % ----------
    % pupil_fpath : Path or cell of paths (all sessions) to pupil data of a subject.
    %
    % Returns
    % -------
    % sts : Status flag.
    opt.mode = 'auto';
    opt.C_z = 625;
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
