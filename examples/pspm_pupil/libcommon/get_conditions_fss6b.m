function [pupil_fpath, conditions] = get_conditions_fss6b(datapath, subj_id)
    pupil_fpath = fullfile(datapath, sprintf('FSS6B_pupil_%.2d.mat', subj_id));
    psych_fpath = fullfile(datapath, sprintf('FSS6B_psych_%.2d.mat', subj_id));
    pupil = load(pupil_fpath);
    psych = load(psych_fpath);

    conditions = struct();
    csp_markers = psych.data(:, 3) == 1;
    csn_markers = psych.data(:, 3) == 2;
    conditions.names = {'CS+', 'CS-'};
    conditions.onsets = {pupil.data{1}.data(csp_markers), pupil.data{1}.data(csn_markers)};
end
