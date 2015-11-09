 function p_d = p_makedata(currPatient, features)
% Make personalized training feature
% Load normalized data
%------------------------------------------------------
% The total number of features is 78. Note:
% 1) 3,4,5 -th feaure has high/low pressure
% 2) 24-28 features only have 2-case values 
% 3) 36,37 features have no distinguishes (2+ for every element)
% 4) 39 has 42 categories
% 5) 50 take < 0.25 as 0.25
%------------------------------------------------------
if ~(exist('features'))
    load features
end

[time_slots, idxA, idxC] = unique(cell2mat(currPatient(:,end)));
% [time_slots, idxA, idxC] = unique(cell2mat(normalized_data(idx, 5)));   % time_slots is already sorted
p_d = ones(length(time_slots), 82)*(-1);     % 78 + 3 (divide pressure
% Add Pain
pain_feature = 'Pain Score';
for iter = 1:79
     switch (iter)
            case {3, 4, 15}
                disp('');
            case {36,37}
                disp('');
            case {39, 40}
                disp('');
            case {50, 51}
                disp('');
            case 79         % Pain data
                feature_rows = find(strcmp(currPatient(:,2), pain_feature)  == 1);
                temp = currPatient(feature_rows,:);         
                noPatientIndex = find(cellfun(@isnumeric, temp(:,3)) ~= 1);
                for i = 1:length(noPatientIndex)
                    temp{noPatientIndex(i), 3} = -1;
                end
                p_d(idxC(feature_rows), iter) = cell2mat(temp(:, 3));
         otherwise
             feature_rows = find(strcmp(currPatient(:,1), features(iter,1)) .* strcmp(currPatient(:,2), features(iter,2)) == 1);
             disp(['Retrive' num2str(iter) 'length: ' num2str(length(feature_rows)) ' ' num2str(length(unique(idxC(feature_rows))))]);
             temp = currPatient(feature_rows,:);         
             p_d(idxC(feature_rows), iter) = cell2mat(temp(:, 3));
     end
end
index = [];
for i = 1:size(p_d, 2)
    if length(find(p_d(:, i) ~= -1)) == 0
        index = [index; i]
    end
end
p_d(:, index) = [];


figure
hold on
for i = 1:size(p_d, 2)
    plot(i, find(p_d(:,i) ~=-1)) 
end

% Add a column which will be used to synchronizing the data
p_d(:,end + 1) = time_slots;

end