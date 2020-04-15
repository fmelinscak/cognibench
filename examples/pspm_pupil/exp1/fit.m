function [out, stats] = fit(inarg)
    out = struct();
    stats = [];
    datapath = inarg.datapath;
    subj_id = inarg.subj_id;
    exclude_segment_length = inarg.exclude_segment_length;
    exclude_cutoff = inarg.exclude_cutoff;

    [pupil_fpath, conditions] = get_conditions(datapath, subj_id);
    [conditions, csp] = single_trial_conditions(conditions);
    channel = 'pupil';
    should_fit = ~isempty(pupil_fpath) && ~isempty(conditions);
    if ~should_fit
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
    options.exclude_missing = struct('segment_length', exclude_segment_length, 'cutoff', exclude_cutoff);

    out = pspm_glm(model, options);
    stats = exclude_and_average(out, csp, out.stats_exclude);
end
