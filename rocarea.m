function [area] = rocarea(roc)
% roc is a matrix with size of [n, 2]
% the first column is TPR, where the second is FPR
% no entry should be greater than 1 and less than 0
if size(roc, 2) == 0
    error('No values recorded');
end
if max(roc(:)) > 1 || min(roc(:)) < 0
    error('Unreasonable TPR or FPR');
end

[v, ind] = sort(roc(:, 1));
sorted = roc(ind, :);

area = 0;
for i = 1:size(roc, 1)
    if i == 1
        area = area + sorted(i, 2) * sorted(i, 1)/2;
    else
        area = area + (sorted(i,2) + sorted(i-1,2))*(sorted(i,1) - sorted(i-1,1))/2;
    end
end

plot(roc(:,1), roc(:,2),'r.')
end