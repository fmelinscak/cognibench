function out = fit(inarg)
    datapath = inarg.datapath;
    subj_id = inarg.subj_id;
    miss_perc_threshold = inarg.miss_perc_threshold;

    [pupil_fpath, conditions] = get_conditions(datapath, subj_id);
    channel = 'pupil';
    should_fit = ~isempty(pupil_fpath) && ~isempty(conditions) && perc_miss_overall(pupil_fpath, channel) <= miss_perc_threshold;
    if ~should_fit
        out = struct();
        return;
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
