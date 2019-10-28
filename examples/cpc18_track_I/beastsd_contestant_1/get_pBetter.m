function [probsBetter] = get_pBetter(DistX, DistY, corr, accuracy)
%Return probability that a value drawn from DistX is strictly larger than one drawn from DistY
  % Input: 2 discrete distributions which are set as matrices of 1st column
  % as outcome and 2nd its probability; correlation between the distributions;
  % level of accuracy in terms of number of samples to take from distributions
  % Output: the estimated probability that X generates value strictly larger than Y, and
  % the probability that Y generates value strictly larger than X

  nXbetter = 0;
  nYbetter = 0;
  for j = 1:accuracy
    rndNum = rand(1);
    sampleX = distSample(DistX(:,1),DistX(:,2),rndNum);
    if corr == 1
      sampleY = distSample(DistY(:,1),DistY(:,2),rndNum);
    elseif corr == -1
      sampleY = distSample(DistY(:,1),DistY(:,2),1-rndNum);
    else
      sampleY = distSample(DistY(:,1),DistY(:,2),rand(1));
    end
    nXbetter = nXbetter + (sampleX > sampleY);
    nYbetter = nYbetter + (sampleY > sampleX);
  end
  pXbetter = nXbetter/accuracy;
  pYbetter = nYbetter/accuracy;
  probsBetter = [pXbetter, pYbetter];
end

