function stats = fit_all(inarg)
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
