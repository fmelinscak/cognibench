function out = fit(datapath, subject_id, bf, foreshortening, preprocessing)
    pupil_fpath = fullfile(datapath, sprintf('FSS6B_pupil_%.2d.mat', subject_id));
    psych_fpath = fullfile(datapath, sprintf('FSS6B_psych_%.2d.mat', subject_id));
    [~, conditions] = read_data(pupil_fpath, psych_fpath);

    model.modelfile = fullfile(datapath, sprintf('glm_%d.mat', subject_id));
    model.datafile = pupil_fpath;
    model.timing = conditions;
    model.timeunits = 'seconds';
    model.modelspec = 'ps_fc';
    model.bf = bf;
    model.channel = 'pupil';
    options.overwrite = true;

    out = pspm_glm(model, options);
end
