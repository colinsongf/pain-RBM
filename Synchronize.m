% Synchronize the data
% The input should be p_d, the output from personalize_data
% 1 Union all the time point 
% 2 Find the overlapped interval for all the interested columns
% 7137, 4822, 1245, 6563
current_ID = 1245
load_name = ['painmatrix' num2str(current_ID) '.mat']
load(load_name)

ts = [];            % timeseries object for all the features (including the pain)
switch current_ID
    case 7137
        interested_cols = [1,2,5,6,7,8,9,10,12,23,25,27,28,29,30,34,35,36,40,42,44,47];                     % 48 is time -- 7137
    case 4822
        interested_cols = [1,2,5,6,7,8,9,10,11,13,23,31,32,34,35,36,40,41,42,45,46,47,49,51,54,57,60,62];   % 63 is time -- 4822
    case 1245
        interested_cols = [1,2,3,4,5,6,7,8,9,10,12,18,19,20,21,22,31,33,35,40,41,45,46,48,52,55,58,59];     % 60 is time -- 1245
    case 6563
        interested_cols = [1,2,4,5,6,7,8,9,10,12,27,29,30,31,34,35,36,40,41,42,46,47,49,51]                   % 52 is time -- 6563
    otherwise
        error('No correspond interested_cols');
end
last_col = interested_cols(end);
used_feature_numbers = length(interested_cols);              % The time is excluded
union_time = [];    % all the time slots
top_limit = 0;
bottom_limit = 100000;
for i = 1:used_feature_numbers
    iter = interested_cols(i);
    rows = find(p_d(:,iter) ~= -1);
    start_row = rows(1);
    end_row = rows(end);
    ts1 = timeseries(p_d(rows, iter), p_d(rows, last_col + 1)); 
    ts = [ts ts1];
    if start_row > top_limit
        top_limit = start_row;
    end
    if end_row < bottom_limit
        bottom_limit = end_row;
    end
    union_time = union(union_time, p_d(rows, last_col + 1));
end 

overlap_start = find(union_time == p_d(top_limit, end));
overlap_end = find(union_time == p_d(bottom_limit, end));
union_time1 = union_time(overlap_start:overlap_end);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pain_indx = find(p_d(:,last_col) ~= -1);
pain_time = ismember(union_time1, p_d(pain_indx, last_col + 1));
unsampled_pain_idx = find(pain_time);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
resampled_ts = [];
test_data_1 = zeros(length(union_time1),used_feature_numbers);
for i = 1:used_feature_numbers
    ts1 = resample(ts(i), union_time1);
    resampled_ts = [resampled_ts ts1];
    test_data_1(:,i) = resampled_ts(i).data;
    if i == used_feature_numbers
        test_data_1(:,i) = test_data_1(:,i)/10;
    end
end

train_data = test_data_1(unsampled_pain_idx, :);       % The training data, with the index of un-resampled pain data
time_sequence = union_time1(unsampled_pain_idx);