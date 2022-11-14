from libsvm.svmutil import *
from liblinear.liblinearutil import csr_scale
import time
# Resources: https://pypi.org/project/libsvm-official/
#            https://github.com/cjlin1/libsvm/tree/master/python

# load data
y, x = svm_read_problem('covtype.libsvm.binary', return_scipy=True)

# pre-proprocess the data by subtracting mean and scale them in [-1,1].
scale_param = csr_find_scale_param(x, lower=-1, upper=1)
x = csr_scale(x, scale_param)

# split to train and test
train_size = int(x.shape[0] * 0.8)

# Use Libsvm, apply kernel methods, see if the accuracy can beat linear SVMs.
prob = svm_problem(y[train_size:], x[train_size:], isKernel=True)
param = svm_parameter('-t 1')

st = time.time()
m = svm_train(prob, param)
et = time.time()
print("Training time applying kernel function: ", et-st)  #602.43 s

# model prediction
p_label, p_acc, p_val = svm_predict(y[train_size:], x[train_size:], m)
ACC, MSE, SCC = evaluations(y[train_size:], p_label)
print("Accuracy: ", round(ACC, 2), "Mean Squared Error: ", round(MSE, 2)) # 59.6448% 0.4
