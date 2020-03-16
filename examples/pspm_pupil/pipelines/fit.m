function out = fit(datapath, subject_id, bf, foreshortening, preprocessing)
    [pupil_fpath, conditions] = get_conditions(datapath, subject_id);

    model.modelfile = fullfile(datapath, sprintf('glm_%d.mat', subject_id));
    model.datafile = pupil_fpath;
    model.timing = conditions;
    model.timeunits = 'seconds';
    model.modelspec = 'ps_fc';
    model.bf = bf;
    model.channel = 'pupil';
    options.overwrite = true;

    out = pspm_glm(model, options);
end

function [pupil_fpath, conditions] = get_conditions(datapath, subject_id)
    fparts = split(datapath, filesep);
    if isempty(fparts{end})
        dataset_name = fparts{end-1};
    else
        dataset_name = fparts{end};
    end
    func = str2func(sprintf('get_conditions_%s', lower(dataset_name)));
    [pupil_fpath, conditions] = func(datapath, subject_id);
end

function [pupil_fpath, conditions] = get_conditions_fss6b(datapath, subject_id)
    pupil_fpath = fullfile(datapath, sprintf('FSS6B_pupil_%.2d.mat', subject_id));
    psych_fpath = fullfile(datapath, sprintf('FSS6B_psych_%.2d.mat', subject_id));
    pupil = load(pupil_fpath);
    psych = load(psych_fpath);

    conditions = struct();
    csp_markers = psych.data(:, 3) == 1;
    csn_markers = psych.data(:, 3) == 2;
    conditions.names = {'CS+', 'CS-'};
    conditions.onsets = {pupil.data{1}.data(csp_markers), pupil.data{1}.data(csn_markers)};
end

function [pupil_fpath, conditions] = get_conditions_vc7b(datapath, subject_id)
    pupil_fpath = fullfile(datapath, sprintf('VC7B_pupil_%.2d_sn1.mat', subject_id));
    psych_fpath = fullfile(datapath, sprintf('VC7B_psych_%.2d.mat', subject_id));
    pupil = load(pupil_fpath);
    psych = load(psych_fpath);

    conditions = struct();
    csp_markers = psych.data(:, 3) == 1;
    csn_markers = psych.data(:, 3) == 2;
    conditions.names = {'CS+', 'CS-'};
    conditions.onsets = {pupil.data{1}.data(csp_markers), pupil.data{1}.data(csn_markers)};
end
