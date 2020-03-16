function stats = fit_all(varargin)
    path_struct = varargin{1};
    args = varargin(2:end);

    datapath = path_struct.datapath;
    subject_ids = path_struct.subject_ids;
    stats = [];
    for subj_id = subject_ids
        glm = fit(datapath, subj_id, args{:});
        stats(1:2, subj_id) = glm.stats(1:2)';
    end
end
