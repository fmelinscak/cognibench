function [out, stats] = fit(inarg)
    out = struct();
    stats = [];
    datapath = inarg.datapath;
    subj_id = inarg.subj_id;
    model_str = inarg.model_str;

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

    if contains(model_str, 'blink_saccade')
        discard_factor = inarg.discard_factor;
        % TODO: need to backup the datasets before running a benchmark
        options.channel_action = 'replace';
        for i = 1:numel(pupil_fpath)
            [sts, ~] = pspm_blink_saccade_filt(pupil_fpath{i}, discard_factor, options);
            if sts ~= 1; return; end;
        end
    elseif contains(model_str, 'pupil_pp')
        options.channel_action = 'replace';
        for i = 1:numel(pupil_fpath)
            [sts, ~] = pspm_pupil_pp(pupil_fpath{i}, options);
            if sts ~= 1; return; end;
        end
    else
        error('ID:invalid_input', '"model_str" must be "blink_saccade" or "pupil_pp"');
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
