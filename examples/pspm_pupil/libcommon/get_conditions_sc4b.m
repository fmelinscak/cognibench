function [pupil_fpath, conditions] = get_conditions_sc4b(datapath, subj_id)
    pupil_fpath = fullfile(datapath, sprintf('SC4B_pupil_%.2d_sn1.mat', subj_id));
    cogent_fpath = fullfile(datapath, sprintf('SC4B_cogent_%.2d.mat', subj_id));
    pupil = load(pupil_fpath);
    cogent = load(cogent_fpath);

    conditions = struct();
    csp_markers = cogent.data(:, 3) == 1;
    csn_markers = cogent.data(:, 3) == 2;
    conditions.names = {'CS+', 'CS-'};
    conditions.onsets = {pupil.data{1}.data(csp_markers), pupil.data{1}.data(csn_markers)};
end
