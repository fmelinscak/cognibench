function out = fit(filepath, bf, foreshortening, preprocessing)
    import{1}.type = 'pupil_l';
    import{1}.eyelink_trackdist = 700;
    import{1}.distance_unit = 'mm';
    import{2}.type = 'pupil_r';
    import{2}.eyelink_trackdist = 700;
    import{2}.distance_unit = 'mm';
    import{3}.type = 'gaze_x_l';
    import{4}.type = 'gaze_x_r';
    import{5}.type = 'gaze_y_l';
    import{6}.type = 'gaze_y_r';
    import{7}.type = 'marker';
    options.overwrite = true;
    pspm_fp = pspm_import(filepath, 'eyelink', import, options);

    model.modelfile = 'fit_glm.mat';
    model.datafile = pspm_fp;
    model.timing.names{1} = 'Cond 0';
    model.timing.names{2} = 'Cond 1';
    model.timing.names{3} = 'Cond 2';
    model.timing.onsets{1} = [1, 4, 8];
    model.timing.onsets{2} = [2, 5, 6];
    model.timing.onsets{3} = [3, 10, 12];
    model.timeunits = 'seconds';
    model.modelspec = 'ps_fc';
    model.bf = bf;
    model.channel = 'pupil_r';
    options.overwrite = true;

    out = pspm_glm(model, options);
end
