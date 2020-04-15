function stats = fit_all(inarg)
    % Fit all the subjects in the dataset given in inarg structure using the currently
    % loaded fit.m function.
    %
    % Before the dataset is fit, it is copied to a temporary location specified with
    % inarg.tmp_out_path so that the original data is left unmodified.
    %
    % Parameters
    % ----------
    % inarg : Structure with the following fields:
    %   .datapath : field holding the path to the dataset to fit.
    %   .tmp_out_path : field holding the path to the temporary location to copy the dataset.
    %   .subject_ids : Array of subject IDs to fit
    %
    % Returns
    % -------
    % stats : Array with shape [2, n_subj] holding CS+ and CS- beta fits for each subject.
    orig_datapath = inarg.datapath;
    ds_name = split(orig_datapath, '/');
    ds_name = ds_name{end};
    tmp_out_path = fullfile(inarg.tmp_out_path, ds_name);
    fprintf('Copying %s to temporary location %s', orig_datapath, tmp_out_path);
    copyfile(orig_datapath, tmp_out_path);
    inarg.datapath = tmp_out_path;
    subject_ids = inarg.subject_ids;
    stats = [];
    for subj_id = subject_ids
        try
            inarg.subj_id = subj_id;
            [glm, cs_stats] = fit(inarg);
            if isempty(fieldnames(glm))
                fprintf('Skipping subject %d...\n', subj_id);
                continue
            end
            stats(1:2, end + 1) = cs_stats;
        catch err
            fprintf('There was an error fitting subject id %d. Skipping...\n', subj_id);
            warning('ID:fitting_error', getReport(err, 'extended'));
        end
    end
    fprintf('Deleting folder %s', tmp_out_path);
    rmdir(tmp_out_path, 's');
end
