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
