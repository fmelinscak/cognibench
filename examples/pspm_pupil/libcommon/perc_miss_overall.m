function perc = perc_miss_overall(fp, channel)
    % Return the overall missing data percentage of the data of a subject in
    % the given channel.
    %
    % Parameters
    % ----------
    % fp : Path or cell to the paths (all sessions) of a subject.
    % channel : Channels on which miss percentage should be computed.
    %
    % Returns
    % -------
    % perc : Overall missing data percentage in the range [0, 100].
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
