function [data, conditions] = read_data(pupil_fpath, psych_fpath)
    pupil = load(pupil_fpath);
    psych = load(psych_fpath);

    data = pupil;

    conditions = struct();
    csp_markers = psych.data(:, 3) == 1;
    csn_markers = psych.data(:, 3) == 2;
    conditions.names = {'CS+', 'CS-'};
    conditions.onsets = {pupil.data{1}.data(csp_markers), pupil.data{1}.data(csn_markers)};
end
