function [pupil_fpath, conditions] = get_conditions_pubfe(datapath, subj_id)
    % Return the pupil paths and timing structure for a subject from PubFe dataset.
    %
    % Parameters
    % ----------
    % datapath : Path to the PubFe dataset.
    % subj_id : ID of the subject
    %
    % Returns
    % -------
    % pupil_fpath : Cell array holding the paths to the pupil files (all sessions) for the
    %               given subject.
    % conditions : Cell array holding the condition structures (all sessions) for the
    %              given subject.
    pupil_fpath = {...
        fullfile(datapath, sprintf('PubFe_pupil_%.2d_sn1.mat', subj_id)),...
        fullfile(datapath, sprintf('PubFe_pupil_%.2d_sn2.mat', subj_id)) ...
    };
    cogent_fpath = fullfile(datapath, sprintf('PubFe_cogent_%.2d.mat', subj_id));
    pupil = {
        load(pupil_fpath{1}),
        load(pupil_fpath{2}),
    };
    n_markers_in_sn1 = numel(pupil{1}.data{1}.data);
    cogent = load(cogent_fpath);

    conditions = {struct(), struct()};
    begs = {0, n_markers_in_sn1, size(cogent.data, 1)};
    for i = 1:2
        sn_markers = cogent.data(begs{i} + 1:begs{i+1}, 3);
        csp_markers = sn_markers == 1;
        csn_markers = sn_markers == 2;
        conditions{i}.names = {'CS+', 'CS-'};
        conditions{i}.onsets = {pupil{i}.data{1}.data(csp_markers), pupil{i}.data{1}.data(csn_markers)};
    end
end
