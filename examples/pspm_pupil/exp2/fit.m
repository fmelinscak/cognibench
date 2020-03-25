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
    [out, stats] = exclude_and_average(out, csp);
end

function [glm, stats] = exclude_and_average(glm, csp)
    csp = cell2mat(csp);
    bfno = glm.bf.bfno;
    stats_csp = 0;
    stats_csn = 0;
    n_csp = 0;
    n_csn = 0;
    for i = 1:numel(csp)
        if glm.stats_exclude(i)
            continue
        end
        idx = (i - 1)*bfno + 1;
        if csp(i)
            stats_csp = stats_csp + glm.stats(idx);
            n_csp = n_csp + 1;
        else
            stats_csn = stats_csn + glm.stats(idx);
            n_csn = n_csn + 1;
        end
    end
    stats_csp = stats_csp / n_csp;
    stats_csn = stats_csn / n_csn;
    stats = [stats_csp stats_csn];
end

function [cond_single, csp] = single_trial_conditions(conditions)
    cond_single = {};
    csp = {};
    trial_cnt = 1;
    for i = 1:numel(conditions)
        cond_single{end + 1} = struct();
        cond_single{end}.names = {};
        cond_single{end}.onsets = [];
        csp_curr = [];
        for j = 1:numel(conditions{i}.names)
            name = conditions{i}.names{j};
            onsets_arr = conditions{i}.onsets{j};
            for k = 1:numel(onsets_arr)
                cond_single{end}.names{end + 1} = sprintf('Trial %d', trial_cnt);
                cond_single{end}.onsets{end + 1} = onsets_arr(k);
                csp_curr(end + 1) = strcmpi(name, 'CS+');
                trial_cnt = trial_cnt + 1;
            end
        end
        csp{end + 1} = csp_curr;
    end
end
