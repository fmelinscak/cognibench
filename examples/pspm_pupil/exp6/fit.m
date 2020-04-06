function [out, stats] = fit(inarg)
    out = struct();
    stats = [];
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
    sts = pp_valid_fixations(pupil_fpath, fixation_angle);
    if sts ~= 1; return; end;

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
