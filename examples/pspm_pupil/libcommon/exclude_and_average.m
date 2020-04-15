function [stats] = exclude_and_average(glm, csp, exclude_arr)
    % Compute CS+ and CS- beta stat averages from a fitted single-trial GLM structure.
    %
    % Parameters
    % ----------
    % glm : Fitted single-trial GLM structure.
    %
    % csp : A logical array where csp(i) is 1 if ith trial is a CS+
    %
    % exclude_arr : A logical array where exclude_arr(i) is 1 if ith trial should be
    %               excluded from the average.
    %
    % Returns
    % -------
    % stats : A 2-element array holding the CS+ and CS- beta averages.
    csp = cell2mat(csp);
    bfno = glm.bf.bfno;
    stats_csp = 0;
    stats_csn = 0;
    n_csp = 0;
    n_csn = 0;
    for i = 1:numel(csp)
        if exclude_arr(i)
            continue
        end
        idx = (i - 1)*bfno + 1;
        if csp(i)
            stats_csp = stats_csp + glm.stats(idx);
            n_csp = n_csp + 1;
        else
            stats_csn = stats_csn + glm.stats(idx);
            n_csn = n_csn + 1;
        end
    end
    stats_csp = stats_csp / n_csp;
    stats_csn = stats_csn / n_csn;
    stats = [stats_csp stats_csn];
end
