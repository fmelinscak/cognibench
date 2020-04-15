function [pupil_fpath, conditions] = get_conditions_doxmem2(datapath, subj_id)
    % Return the pupil paths and timing structure for a subject from DoxMem2 dataset.
    %
    % Parameters
    % ----------
    % datapath : Path to the doxmem2 dataset.
    % subj_id : ID of the subject
    %
    % Returns
    % -------
    % pupil_fpath : Cell array holding the paths to the pupil files (all sessions) for the
    %               given subject.
    % conditions : Cell array holding the condition structures (all sessions) for the
    %              given subject.
    cogent_fpath = {...
        fullfile(datapath, sprintf('DoxMem2_cogent_%.2d_sn1.mat', subj_id)),
        fullfile(datapath, sprintf('DoxMem2_cogent_%.2d_sn2.mat', subj_id)),
        fullfile(datapath, sprintf('DoxMem2_cogent_%.2d_sn3.mat', subj_id)),
    };
    cogent = {
        load(cogent_fpath{1}),
        load(cogent_fpath{2}),
        load(cogent_fpath{3}),
    };
    if cogent{1}.subject.drug == 1
        pupil_fpath = {};
        conditions = {};
        return;
    end
    pupil_fpath = {...
        fullfile(datapath, sprintf('DoxMem2_pupil_%.2d_sn1.mat', subj_id)),...
        fullfile(datapath, sprintf('DoxMem2_pupil_%.2d_sn2.mat', subj_id)) ...
        fullfile(datapath, sprintf('DoxMem2_pupil_%.2d_sn3.mat', subj_id)) ...
    };
    pupil = {
        load(pupil_fpath{1}),
        load(pupil_fpath{2}),
        load(pupil_fpath{3}),
    };
    n_markers_in_sn1 = numel(pupil{1}.data{1}.data);
    n_markers_in_sn2 = numel(pupil{2}.data{1}.data);

    conditions = {struct(), struct(), struct()};
    for i = 1:3
        sn_markers = cogent{i}.data(:, 3);
        csp_markers = (sn_markers == 2) | (sn_markers == 3);
        csn_markers = sn_markers == 1;
        conditions{i}.names = {'CS+', 'CS-'};
        conditions{i}.onsets = {pupil{i}.data{1}.data(csp_markers), pupil{i}.data{1}.data(csn_markers)};
    end
end
