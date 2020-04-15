function [pupil_fpath, conditions] = get_conditions_fss6b(datapath, subj_id)
    % Return the pupil paths and timing structure for a subject from FSS6B dataset.
    %
    % Parameters
    % ----------
    % datapath : Path to the FSS6B dataset.
    % subj_id : ID of the subject
    %
    % Returns
    % -------
    % pupil_fpath : Path to the pupil file for the given subject.
    % conditions : Condition structure for the given subject.
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
