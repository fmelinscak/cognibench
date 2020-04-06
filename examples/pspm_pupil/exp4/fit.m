function [out, stats] = fit(inarg)
    model_str = inarg.model_str;
    if strcmpi(model_str, 'single-trial')
        [out, stats] = single_trial(inarg);
    elseif strcmpi(model_str, 'condition-wise')
        [out, stats] = condition_wise(inarg);
    else
        error('ID:invalid_input', '"model_str" must be "single-trial" or "condition-wise"');
    end
end

function [out, stats] = single_trial(inarg)
    out = struct();
    stats = [];
    datapath = inarg.datapath;
    subj_id = inarg.subj_id;
    trial_exclusion_threshold = inarg.trial_exclusion_threshold;
    exclusion_seg_len = inarg.exclude_segment_length;
    participant_exclusion_threshold = inarg.participant_exclusion_threshold;

    [pupil_fpath, conditions] = get_conditions(datapath, subj_id);
    [conditions, csp] = single_trial_conditions(conditions);
    channel = 'pupil';
    should_fit = ~isempty(pupil_fpath) && ~isempty(conditions) && perc_miss_overall(pupil_fpath, channel) <= participant_exclusion_threshold;
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
    options.exclude_missing = struct('segment_length', exclusion_seg_len, 'cutoff', trial_exclusion_threshold);

    out = pspm_glm(model, options);
    [out, stats] = exclude_and_average(out, csp, out.stats_exclude);
end

function [out, stats] = condition_wise(inarg)
    out = struct();
    stats = [];
    datapath = inarg.datapath;
    subj_id = inarg.subj_id;
    participant_exclusion_threshold = inarg.participant_exclusion_threshold;

    [pupil_fpath, conditions] = get_conditions(datapath, subj_id);
    channel = 'pupil';
    should_fit = ~isempty(pupil_fpath) && ~isempty(conditions) && perc_miss_overall(pupil_fpath, channel) <= participant_exclusion_threshold;
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

    out = pspm_glm(model, options);
    bfno = out.bf.bfno;
    stats = [out.stats(1) out.stats(1 + bfno)];
end
