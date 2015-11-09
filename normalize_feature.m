function [outputs] = normalize_feature(data)
% Normalize the data (specific patient or total data) based on the probability of the state is
% normal/abnormal
% normal: probability ~ 0
% abnorma: pro1bability ~ 1
% Currently, the 3,4,1,39,40,50,51 features are not normalized
% 36,37 features will not be used always
if ~exist('features', 'var')
    load features
end
% load features.mat
normalized_data = data;
avaliable_index = [1:78];
special_index = [3,4,15,24,25,26,27,28,36,37,39,40,50,51];

avaliable_index(special_index) = [];

for iter = 1:length(avaliable_index)
    i = avaliable_index(iter);
    indx = find(strcmp(data(:,2), features(i,1)) .* strcmp(data(:,3), features(i,2)) == 1);
    temp = [data{indx, 4}];
    meanX = mean(temp);
    stdX = std(temp);
    prob = 1 - normpdf(temp, meanX, stdX) * sqrt(2 * pi) * stdX;
    normalized_data(indx, 4) = num2cell(prob);
    temp = [];
end

% Binary string data
for i = 24:28
    t0_string = 'Less than/equal to 3 seconds';
    t1_string = 'Greater than 3 seconds';
    if i == 28
        t0_string = 'Less Than 2 Seconds';
        t1_string = 'Greater Than 2 Seconds';
    end
    indx = find(strcmp(data(:,2), features(i,1)) .* strcmp(data(:,3), features(i,2)) == 1);
    t0 = find (cellfun(@isempty, strfind(data(indx, 4), t0_string)) == 0);
    t1 = find (cellfun(@isempty, strfind(data(indx, 4), t1_string)) == 0);
    normalized_data(indx(t0), 4) = num2cell(zeros(length(t0), 1));
    normalized_data(indx(t1), 4) = num2cell(ones(length(t1), 1));
end

outputs = normalized_data;
% save('normalized_data.mat', 'normalized_data');
end