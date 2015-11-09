%------------------
% datapre.m
% First step:
% Convert the raw data, Remove the element of non-desirable
%------------------
load pain_data.mat
data = raw;        
data(:, 2) = [];        % delete the useless feature
noValue = strcmp(data(:,4), 'NULL');
data(noValue, :) = [];
noValue = strcmp(data(:,4),'Not assessed');
data(noValue, :) = [];
noValue = strcmp(data(:, 4), 'NA (pre med)');
data(noValue, :) = [];
noValue = strcmp(data(:, 4), 'NA (fever)');
data(noValue, :) = [];
noTime = find(cellfun(@isempty, strfind(data(:,5), 'NULL')) == 0);
data(noTime, :) = [];   % delete the null time
% data(2:end, 5) = num2cell(datenum(data(2:end, 5)) + 1990);  % date->number + a constant
[pID, IA, IC] = unique(data(:,1));
data(:,1) = num2cell(IC);
save('data.mat', 'data', 'pID', 'IA', 'IC');