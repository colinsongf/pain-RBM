% make the specific patient data
% e.g: personalize_data(4822)

% Next: Synchronize
% PID_number = 7137, 4822, 1245, 6563
function [p_d] = personalize_data(PID_number)
    if ~exist('data', 'var')
        load 'data.mat'
    end
    if ~exist('features', 'var')
        load 'features.mat'
    end
    %PID_number = 7137;
    idx = find(cell2mat(data(:,1)) == PID_number);
    currPatient = normalize_feature(data(idx, :));
    currPatient(:, 5) = num2cell(datenum(currPatient(:, 5)) + 1990);
    p_d = p_makedata(currPatient(:,2:5), features);
    save_name = ['painmatrix' num2str(PID_number) '.mat'];
    save(save_name, 'p_d');
    % 
end