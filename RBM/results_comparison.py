import os
import numpy as np
import scipy.io as sio
from sklearn import svm
from sklearn.decomposition import PCA, KernelPCA
from sklearn.lda import LDA
from nolearn.dbn import DBN


def ROCvalue(predict_labels, real_labels):

	assert len(predict_labels) == len(real_labels), 'The length of predict and real labels are not idential'

	TP, TN, FP, FN = [0, 0, 0, 0]
	for i in xrange(len(real_labels)):
		if real_labels[i] == 0:
			if predict_labels[i] == 0:
				TN += 1
			else:
				FP += 1
		else:
			if predict_labels[i] == 0:
				FN += 1
			else:
				TP += 1
	
	TPR = (float(TP) / (TP + FN))
	FPR = (float(FP) / (FP + TN))

	print 'FPR, TPR are: ', FPR, TPR
	print 'The distance from (0,1) is ', (1-TPR)*(1-TPR) + FPR * FPR
	print 'Positive Likelihood Ratio: ', TPR / FPR
	print 'Accuracy is: ', (TP + TN)/ float(len(real_labels))

	return TPR / FPR

def ROCarea(*args):
	predict_prob = args[0]
	real_labels = args[1]
	thresh = args[2]
	record_flag = args[3]
	if record_flag == True: 
		recorded_label = np.invert(predict_prob[:,0] >= thresh)

	assert predict_prob.shape[0] == len(real_labels), 'The length of predict and real labels are not identical'
	TP, TN, FP, FN = [0, 0, 0, 0] 
	for i in xrange(len(real_labels)):
		if real_labels[i] == 0:
			if predict_prob[i,0] >= thresh:
				TN += 1
			else:
				FP += 1
		else:
			if predict_prob[i,0] >= thresh:
				FN += 1
			else:
				TP += 1
	TPR, FPR = [(float(TP) / (TP + FN)), (float(FP) / (FP + TN))] 
	print 'TP, TN, FP, FN, FPR, TPR are: ', TP, TN, FP, FN, FPR, TPR
	
	if record_flag == True:
		print recorded_label
		return (FPR, TPR), recorded_label.astype(int)
	else:
		return FPR, TPR



current_ID = 6563   
current_folder = str(current_ID)

if not os.path.exists(current_folder):
	os.makedirs(current_folder)

#---------------------Snippet 1, read mat file
data = sio.loadmat('../train_data_'+str(current_ID)+'_test.mat')
data_end_col = data['train_data'].shape[1]
train_data = data['train_data'][:,0:data_end_col - 1]
test_data = data['test_data'][:,0:data_end_col - 1]

train_labels = data['train_data'][:, data_end_col - 1]
test_labels = data['test_data'][:, data_end_col - 1]
#---------------------End of Snippet

print train_data.shape
print test_data.shape
#---------------------Snippet 2, perform PCA, and LDA
pca = PCA(n_components = 15)
pca.fit(train_data)
PCA_train_data = pca.transform(train_data)
PCA_test_data = pca.transform(test_data)

lda_clf = LDA()			# No difference when set the n_componenets, since n_class - 1 = 1, therefore, only one dimension was left
lda_clf.fit(train_data, train_labels)


# print (pca.explained_variance_ratio_)
#---------------------End of Snippet

#---------------------Snippet 3, PCA followed by a linear SVM classification

# Using PCA
pca_clf = svm.SVC(probability = True)
pca_clf.fit(PCA_train_data, train_labels)
pca_pResults = pca_clf.predict(PCA_test_data)
pca_pResults_prob = pca_clf.predict_proba(PCA_test_data)

print pca_pResults 
# Using LDA
lda_pResults = lda_clf.predict(test_data)
lda_pResults_prob = lda_clf.predict_proba(test_data)

#---------------------Check the order of the classes, make sure we use it correctly
print 'the class order of lda is: ', lda_clf.classes_
print 'the class order of pca is: ', pca_clf.classes_

#---------------------Training of DBN
'''
dbn = DBN(
	[train_data.shape[1], 30, 30,2],
	learn_rates = 0.35,
	learn_rate_decays = 1,
	epochs = 30,
	verbose = 1,
	dropouts = 0.04,
	)

dbn.fit(train_data, train_labels)

dbn_pResults = dbn.predict(test_data)
'''

#---------------------End of Snippet
# rocPCA  = ROCvalue(pca_pResults, test_labels)

rocLDA  = ROCvalue(lda_pResults, test_labels)
# rocDBN = ROCvalue(dbn_pResults, test_labels)

#sio.savemat('LINEARroc_test.mat', mdict = {'LDA_record_labels':lda_pResults, 'PCA_record_labels':pca_pResults, 'DBN_record_labels':dbn_pResults})


LDA_record_labels = [False] * len(test_labels)
PCA_record_labels = [False] * len(test_labels)
PCAroc = np.ones((80, 2))
LDAroc = np.ones((80, 2))
LDA_record_number = 46		
PCA_record_number = 42

print pca_pResults_prob
print lda_pResults_prob
for i in xrange(80):
	thresh = i/80.0
	print thresh
	LDAroc[i,:] = ROCarea(lda_pResults_prob, test_labels, thresh, False)
	PCAroc[i,:] = ROCarea(pca_pResults_prob, test_labels, thresh, False)
	if i == LDA_record_number:
		LDAroc[i,:], LDA_record_labels = ROCarea(lda_pResults_prob, test_labels, thresh, True)
	if i == PCA_record_number:
		PCAroc[i,:], PCA_record_labels = ROCarea(pca_pResults_prob, test_labels, thresh, True)

	
save_name = './'+str(current_ID)+'/LINEARroc.mat'
sio.savemat(save_name, mdict = {'PCAroc':PCAroc, 'LDAroc':LDAroc, 'LDA_record_labels':LDA_record_labels, 'PCA_record_labels':PCA_record_labels, 'test': False})
print LDA_record_labels
print PCA_record_labels
