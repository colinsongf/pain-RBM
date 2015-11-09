%%
% Plotting the ROC curve, mainly compare RBM, PCA, and LDA
% 
% 7137, 4822, 1245, 6563
current_ID = 6563;
file_folder =  num2str(current_ID);
hf = figure;
hold on;
RBM_file = ['./' num2str(current_ID) '/RBMroc.mat'];
load(RBM_file);
if exist('RBMroc', 'var')
    RBMarea = plotROC(RBMroc, 'r', '*', [.5 .7 .5]);
    legend(['RBM' ' with AUC = ' num2str(RBMarea)]);
end

LINEAR_file = ['./' num2str(current_ID) '/LINEARroc.mat'];
load(LINEAR_file)
PCAarea = plotROC(PCAroc, 'g', 'o', [.5 .5 .9]);
LDAarea = plotROC(LDAroc, 'm', '+', [.9 .5 .5]);


title(['ROC Curve of Patient #' num2str(current_ID)]);
xlabel('False Positive Rate');
ylabel('True Positive Rate');

legend(['RBM' ' with AUC = ' num2str(RBMarea)], ['PCA' ' with AUC = ' num2str(PCAarea)], ['LDA' ' with AUC = ' num2str(LDAarea)], 'location', 'southeast');
plot([0 1], [0 1], 'k:')
