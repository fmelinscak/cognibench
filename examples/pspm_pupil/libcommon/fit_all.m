function stats = fit_all(inarg)
    datapath = inarg.datapath;
    subject_ids = inarg.subject_ids;
    stats = [];
    for subj_id = subject_ids
        try
            inarg.subj_id = subj_id;
            glm = fit(inarg);
            if isempty(fieldnames(glm))
                fprintf('Skipping subject %d...\n', subj_id);
                continue
            end
            stats(1:2, end + 1) = glm.stats(1:2)';
        catch err
            fprintf('There was an error fitting subject id %d. Skipping...\n', subj_id);
            warning('ID:fitting_error', getReport(err, 'extended'));
        end
    end
end
