function out = fit(inarg)
    out = struct();
    datapath = inarg.datapath;
    subj_id = inarg.subj_id;
    fixation_angle = inarg.fixation_angle;

    [pupil_fpath, conditions] = get_conditions(datapath, subj_id);
    channel = 'pupil';
    should_fit = ~isempty(pupil_fpath) && ~isempty(conditions);
    if ~should_fit
        return;
    end

    if ~iscell(pupil_fpath)
        pupil_fpath = {pupil_fpath};
    end
    fprintf('Processing subject %d\n', subj_id);
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

    model.modelfile = fullfile(datapath, sprintf('glm_%d.mat', subj_id));
    model.datafile = pupil_fpath;
    model.timing = conditions;
    model.timeunits = 'seconds';
    model.modelspec = 'ps_fc';
    bf.fhandle = @pspm_bf_psrf_fc;
    bf.args = [1, 1, 0];
    model.bf = bf;
    model.channel = channel;
    options.overwrite = true;

    out = pspm_glm(model, options);
end
