function stats = fit_all(varargin)
    path_struct = varargin{1};
    args = varargin(2:end);

    datapath = path_struct.datapath;
    subject_ids = path_struct.subject_ids;
    stats = [];
    for subj_id = subject_ids
        try
            glm = fit(datapath, subj_id, args{:});
            if isempty(fieldnames(glm))
                fprintf('Skipping subject %d...\n', subj_id);
                continue
            end
            stats(1:2, end + 1) = glm.stats(1:2)';
        catch
            fprintf('There was an error fitting subject id %d. Skipping...\n', subj_id);
        end
    end
end
