function [pupil_fpath, conditions] = get_conditions_fer02(datapath, subj_id)
    cogent_fpath = {};
    cogent = {};
    try
        for i = 1:3
            path = fullfile(datapath, sprintf('FER02_cogent_%.2d_sn%d.mat', subj_id, i));
            cogent{end + 1} = load(path);
            cogent_fpath{end + 1} = path;
        end
    catch
        fprintf('Subject %d doesn''t have all three sessions\n', subj_id);
    end
    pupil_fpath = {};
    pupil = {};
    try
        for i = 1:3
            path = fullfile(datapath, sprintf('FER02_pupil_%.2d_sn%d.mat', subj_id, i));
            pupil{end + 1} = load(path);
            pupil_fpath{end + 1} = path;
        end
    catch
    end

    n_sess = numel(pupil_fpath);
    conditions = {};
    for i = 1:n_sess
        conditions{end + 1} = struct();
        sn_markers = cogent{i}.data(:, 3);
        csp_markers = (sn_markers == 2) | (sn_markers == 3);
        csn_markers = sn_markers == 1;
        conditions{i}.names = {'CS+', 'CS-'};
        conditions{i}.onsets = {pupil{i}.data{1}.data(csp_markers), pupil{i}.data{1}.data(csn_markers)};
    end
end
