% check_pain_labels.m
% Sceond step:
% Check the patient data with good pain labels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~exist('IA', 'var')
    load 'data.mat'
end
Total = zeros(length(IA), 1);       % the number of recorded pain data (from 0 to 10)
Pain = zeros(length(IA), 1);        % the number of recorded pain data (>1)
parfor PID_number = 1:length(IA)
    idx = find(cell2mat(data(:,1)) == PID_number);
    pain_idx = strfind(data(idx, 3), 'Pain Score');
    pain_Index = find(not(cellfun('isempty', pain_idx)));
    L = length(pain_Index);
    painNumber = length(find(cell2mat(data(idx(pain_Index),4)) > 0));
    Total(PID_number) = L;
    Pain(PID_number) = painNumber;
end
save('painlabels.mat', 'Pain', 'Total');