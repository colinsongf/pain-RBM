% Retrive training features
%------------------------------------------------------
% The total number of features is 78. Note:
% 1) 3,4,5 -th feaure has high/low pressure
% 2) 24-28 features only have 2-case values 
% 3) 36,37 features have no distinguishes (2+ for every element)
% 4) 39 has 42 categories
% 5) 50 take < 0.25 as 0.25
% 6) 51 currently nothing
%------------------------------------------------------
function single_time = retrive_features(time, currPatient, features)
    for kth_feature = 1:78
        switch (kth_feature)
            case {3, 4, 5}
                disp;
            case {36,37}
                disp;
            case {39, 40}
                disp('Will not be retrived, wait for response');
            case {50, 51}
                disp('');
            otherwise
                row_number2 = find(strcmp(currPatient(:,1), features(i,1)) .* strcmp(currPatient(:,2), features(i,2)) == 1);
                single_feature(kth-feature) = currPatient(row_number, 3);
                disp('Standarlize dthe probability compute');
        end
    end
end

