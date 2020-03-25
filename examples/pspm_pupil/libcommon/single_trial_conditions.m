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
