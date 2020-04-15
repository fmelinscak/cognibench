function [pupil_fpath, conditions] = get_conditions(datapath, subj_id)
    % Return the pupil paths and timing structure for a subject. The type of the
    % dataset is inferred from datapath.
    %
    % Parameters
    % ----------
    % datapath : Path to the dataset.
    % subj_id : ID of the subject
    %
    % Returns
    % -------
    % pupil_fpath : Path or cell array holding the paths to the pupil files (all sessions)
    %               for the given subject.
    % conditions : Struct or cell array holding the condition structures (all sessions)
    %              for the given subject.
    fparts = split(datapath, filesep);
    if isempty(fparts{end})
        dataset_name = fparts{end-1};
    else
        dataset_name = fparts{end};
    end
    func = str2func(sprintf('get_conditions_%s', lower(dataset_name)));
    [pupil_fpath, conditions] = func(datapath, subj_id);
end
