function [ Dist ] = CPC18_getDist( H, pH, L, LotShape, LotNum )
% Extract true full distributions of an option in CPC18
%   input is high outcome (int), its probability (double), low outcome
%   (int), the shape of the lottery ('-'/'Symm'/'L-skew'/'R-skew' only), and
%   the number of outcomes in the lottery
%   output is a matrix with first column a list of outcomes (sorted
%   ascending) and the second column their respective probabilities.

 if strcmp(LotShape, '-')
     if  pH == 1
         Dist = [H pH];
     else
         Dist = [L, 1-pH; H, pH];
     end
 else % H is multioutcome
     %compute H distribution
     if strcmp(LotShape,'Symm')
         highDist = NaN(LotNum, 2);
         k = LotNum - 1;
         for i = 0:k
             highDist(i+1,1) = H - k/2 + i;
             highDist(i+1,2) = pH * binopdf(i,k,0.5);
         end
     elseif strcmp(LotShape,'R-skew') || strcmp(LotShape,'L-skew')
         highDist = NaN(abs(LotNum),2);
         if strcmp(LotShape,'R-skew') > 0
             C = -1-LotNum;
             distsign = 1;
         else
             C = 1+LotNum;
             distsign = -1;
         end
         for i = 1:LotNum
             highDist(i,1) = H + C + distsign*2^i;
             highDist(i,2) = pH/(2^i);
         end
         highDist(LotNum,2) = highDist(LotNum,2)*2;
     end
     % incorporate L into the distribution
     Dist = highDist;
     [lia, locb] = ismember(L,highDist(:,1));
     if lia
         Dist(locb,2) = Dist(locb,2) + (1-pH);
     elseif pH < 1
         Dist(size(Dist,1)+1,:) = [L 1-pH];
     end
     Dist = sortrows(Dist);
 end

end

