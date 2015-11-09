clear
close;

load 'LINEARroc_test.mat';
%% Simply for find the time for patient 7137
% This time should be recored
% load train_data_71373.mat
load train_data_7137_test_time
test = test_data;
load train_data_7137.mat
original = train_data;
load 7137time

back_indx = [];
for i = 1:length(test)
    [~, indxx] = ismember(test(i,1:end-1), original(:,1:end-1), 'rows');
    back_indx(i) = indxx;
end

[timelabels, idx] = sort(time_sequence(back_indx));

started_time = 119;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% True Labels
true_label = test(:,end);
len = length(true_label); 
x_axis = timelabels(started_time:end);

tc_true_label = true_label(idx);                %tc for time_corrected
tc_true_label = tc_true_label(started_time:end)
hf = figure;
o_size = get(hf, 'Position');
set(hf, 'Position', [100 100 700 500]);
% set(hf, 'Position', [o_size(1) o_size(2)-200 o_size(3)+50 o_size(4)+50])
ax = axes('Parent',hf,...
                  'Units','pixels',...
                  'Tag','ax');
y = get(ax, 'Position');
set(ax, 'Position', [y(1)+20 y(2)+50 y(3) y(4)+100])
hold on;
pain_spots = (tc_true_label == 1)';
nopain_spots = (tc_true_label == 0)';
stem(x_axis(pain_spots), tc_true_label(pain_spots), 'r');
stem(x_axis(nopain_spots), tc_true_label(nopain_spots), 'b');

% DBN Labels
base = 2;
tc_dbn_label = DBN_record_labels(idx);
tc_dbn_label = tc_dbn_label(started_time:end)
dbn_pain_spots = (tc_dbn_label == 1);
dbn_nopain_spots = (tc_dbn_label == 0);

sh1 = stem(x_axis(pain_spots & dbn_pain_spots), tc_dbn_label(pain_spots & dbn_pain_spots) + base, 'r', 'BaseValue', base, 'MarkerFaceColor','red','Marker','d');
sh2 = stem(x_axis(dbn_pain_spots & nopain_spots), (tc_dbn_label(dbn_pain_spots & nopain_spots)) + base, 'g', 'BaseValue', base, 'MarkerFaceColor','g','Marker','d')
sh3 = stem(x_axis(nopain_spots & dbn_nopain_spots), tc_dbn_label(nopain_spots & dbn_nopain_spots) + base, 'b', 'BaseValue', base, 'MarkerFaceColor','b','Marker','d');
sh4 = stem(x_axis(pain_spots & dbn_nopain_spots), tc_dbn_label(pain_spots & dbn_nopain_spots) + base, 'k', 'BaseValue', base, 'MarkerFaceColor','k','Marker','d');
% 

% PCA Labels
base = 4;
tc_PCA_label = PCA_record_labels(idx);
tc_PCA_label = tc_PCA_label(started_time:end)
pca_pain_spots = (tc_PCA_label == 1);
pca_nopain_spots = (tc_PCA_label == 0);

sh5 = stem(x_axis(pain_spots & pca_pain_spots), tc_PCA_label(pain_spots & pca_pain_spots) + base, 'r', 'BaseValue', base,'MarkerFaceColor','red','Marker','v');
sh6 = stem(x_axis(pca_pain_spots & nopain_spots), (tc_PCA_label(pca_pain_spots & nopain_spots)) + base, 'g', 'BaseValue', base, 'MarkerFaceColor','g','Marker','v');
sh7 = stem(x_axis(nopain_spots & pca_nopain_spots), tc_PCA_label(nopain_spots & pca_nopain_spots) + base, 'b', 'BaseValue', base, 'MarkerFaceColor','b','Marker','v');
sh8 = stem(x_axis(pain_spots & pca_nopain_spots), tc_PCA_label(pain_spots & pca_nopain_spots) + base, 'k', 'BaseValue', base, 'MarkerFaceColor','k','Marker','v');

baseline_handle = get(sh8, 'Baseline');
set(baseline_handle,'LineStyle','--','Color','k')
% set(baseline_handle,'Visible','off')
% LDA Labels
base = 6;
tc_LDA_label = LDA_record_labels(idx);
tc_LDA_label = tc_LDA_label(started_time:end);
lda_pain_spots = (tc_LDA_label == 1);
lda_nopain_spots = (tc_LDA_label == 0);

stem(x_axis(pain_spots & lda_pain_spots), tc_LDA_label(pain_spots & lda_pain_spots) + base, 'r', 'BaseValue', base,'MarkerFaceColor','red','Marker','s');
stem(x_axis(lda_pain_spots & nopain_spots), (tc_LDA_label(lda_pain_spots & nopain_spots)) + base, 'g', 'BaseValue', base,'MarkerFaceColor','g','Marker','s');
stem(x_axis(nopain_spots & lda_nopain_spots), tc_LDA_label(nopain_spots & lda_nopain_spots) + base, 'b', 'BaseValue', base,'MarkerFaceColor','b','Marker','s');
stem(x_axis(pain_spots & lda_nopain_spots), tc_LDA_label(pain_spots & lda_nopain_spots) + base, 'k', 'BaseValue', base,'MarkerFaceColor','k','Marker','s');


axis([0 200 0 8])
title('Pain prediction results using PCA, LDA and DBN for patient 7137', 'fontsize', 60);

Xl = [min(x_axis), max(x_axis)];


Xt = linspace(Xl(1), Xl(2), 25);
set(gca,'XTickLabel','')
set(gca, 'YTickLabel','')
set(gca,'XTick',Xt,'XLim',Xl);
plot(Xl, [4 4],'LineStyle','--','Color','k');
plot(Xl, [2 2],'LineStyle','--','Color','k');
ax = axis;    % Current axis limits
axis(axis);    % Set the axis limit modes (e.g. XLimMode) to manual
Yl = ax(3:4);  % Y-axis limits
% Place the text labels

t = text(Xt,Yl(1)*ones(1,length(Xt)), datestr(Xt));
set(t,'HorizontalAlignment','right','VerticalAlignment','top', ...
      'Rotation',45, 'fontsize', 7);

set(gca,'XMinorTick','on','YMinorTick','on')

set(gca,'xgrid','on')
% Remove the default labels
legend('Real positive', 'Real negative', ...
        'DBN true positive', 'DBN false positive', 'DBN true nagetive', 'DBN false negative', ...
         'PCA true positive', 'PCA false positive', 'PCA true nagetive', 'PCA false negative',...
          'LDA true positive', 'LDA false positive', 'LDA true nagetive', 'LDA false negative');

 createtextbox1(hf)
 createtextbox2(hf)
 createtextbox3(hf)
 createtextbox4(hf)