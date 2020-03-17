function out = fit(datapath, subject_id, miss_perc_threshold)
    [pupil_fpath, conditions] = get_conditions(datapath, subject_id);
    channel = 'pupil';
    should_fit = ~isempty(pupil_fpath) && ~isempty(conditions) && perc_miss_overall(pupil_fpath, channel) <= miss_perc_threshold;
    if ~should_fit
        out = struct();
        return;
    end

    model.modelfile = fullfile(datapath, sprintf('glm_%d.mat', subject_id));
    model.datafile = pupil_fpath;
    model.timing = conditions;
    model.timeunits = 'seconds';
    model.modelspec = 'ps_fc';
    bf.fhandle = @pspm_bf_psrf_fc;
    bf.args = [1, 1, 0];
    model.bf = bf;
    model.channel = channel;
    options.overwrite = true;

    out = pspm_glm(model, options);
end

function perc = perc_miss_overall(fp, channel)
    if ~iscell(fp)
        fp = {fp};
    end
    n_miss = 0.0;
    n_total = 0.0;
    for i = 1:numel(fp)
        [sts, ~, data] = pspm_load_data(fp{i}, channel);
        arr = data{end}.data;
        n_miss = n_miss + sum(isnan(arr));
        n_total = n_total + numel(arr);
    end
    perc = 100 * (n_miss / n_total);
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

function [pupil_fpath, conditions] = get_conditions_sc4b(datapath, subject_id)
    pupil_fpath = fullfile(datapath, sprintf('SC4B_pupil_%.2d_sn1.mat', subject_id));
    cogent_fpath = fullfile(datapath, sprintf('SC4B_cogent_%.2d.mat', subject_id));
    pupil = load(pupil_fpath);
    cogent = load(cogent_fpath);

    conditions = struct();
    csp_markers = cogent.data(:, 3) == 1;
    csn_markers = cogent.data(:, 3) == 2;
    conditions.names = {'CS+', 'CS-'};
    conditions.onsets = {pupil.data{1}.data(csp_markers), pupil.data{1}.data(csn_markers)};
end

function [pupil_fpath, conditions] = get_conditions_li(datapath, subject_id)
    pupil_fpath = {...
        fullfile(datapath, sprintf('LI_pupil_%.2d_sn1.mat', subject_id)),...
        fullfile(datapath, sprintf('LI_pupil_%.2d_sn2.mat', subject_id)) ...
    };
    cogent_fpath = fullfile(datapath, sprintf('LI_cogent_%.2d.mat', subject_id));
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

function [pupil_fpath, conditions] = get_conditions_pubfe(datapath, subject_id)
    pupil_fpath = {...
        fullfile(datapath, sprintf('PubFe_pupil_%.2d_sn1.mat', subject_id)),...
        fullfile(datapath, sprintf('PubFe_pupil_%.2d_sn2.mat', subject_id)) ...
    };
    cogent_fpath = fullfile(datapath, sprintf('PubFe_cogent_%.2d.mat', subject_id));
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

function [pupil_fpath, conditions] = get_conditions_doxmem2(datapath, subject_id)
    cogent_fpath = {...
        fullfile(datapath, sprintf('DoxMem2_cogent_%.2d_sn1.mat', subject_id)),
        fullfile(datapath, sprintf('DoxMem2_cogent_%.2d_sn2.mat', subject_id)),
        fullfile(datapath, sprintf('DoxMem2_cogent_%.2d_sn3.mat', subject_id)),
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
        fullfile(datapath, sprintf('DoxMem2_pupil_%.2d_sn1.mat', subject_id)),...
        fullfile(datapath, sprintf('DoxMem2_pupil_%.2d_sn2.mat', subject_id)) ...
        fullfile(datapath, sprintf('DoxMem2_pupil_%.2d_sn3.mat', subject_id)) ...
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
