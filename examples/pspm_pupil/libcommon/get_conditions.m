function [pupil_fpath, conditions] = get_conditions(datapath, subj_id)
    fparts = split(datapath, filesep);
    if isempty(fparts{end})
        dataset_name = fparts{end-1};
    else
        dataset_name = fparts{end};
    end
    func = str2func(sprintf('get_conditions_%s', lower(dataset_name)));
    [pupil_fpath, conditions] = func(datapath, subj_id);
end
