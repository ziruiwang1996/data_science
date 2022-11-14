from liblinear.liblinearutil import *
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import time
# Resources: https://github.com/cjlin1/liblinear/tree/master/python
#            https://pypi.org/project/libsvm-official/

# load data
y, x = svm_read_problem('covtype.libsvm.binary', return_scipy=True)

# pre-proprocess the data by subtracting mean and scale them in [-1,1].
scale_param = csr_find_scale_param(x, lower=-1, upper=1)
x = csr_scale(x, scale_param)

# split to train and test
train_size = int(x.shape[0] * 0.8)

# Use Liblinear, apply SVM (primal), SVM (dual) and compare results and running time.
st_p = time.time()
m_primal = train(y[:train_size], x[:train_size], '-s 2')
et_p = time.time()
st_d = time.time()
m_dual = train(y[:train_size], x[:train_size], '-s 1')
et_d = time.time()
print("Primal Training Time: ", et_p-st_p)  #4.91s
print("Dual Training Time: ", et_d-st_d)    #63.76s

# Use Libsvm, apply kernel methods, see if the accuracy can beat linear SVMs.
# Code in p3_libsvm.py

# test primal and dual models
p_label_p, p_acc_p, p_val_p = predict(y[train_size:], x[train_size:], m_primal)
p_label_d, p_acc_d, p_val_d = predict(y[train_size:], x[train_size:], m_dual)
ACC_p, MSE_p, SCC_p = evaluations(y[train_size:], p_label_p)
ACC_d, MSE_d, SCC_d = evaluations(y[train_size:], p_label_d)
print("Primal: ")
print("Accuracy: ", round(ACC_p, 2), "Mean Squared Error: ", round(MSE_p, 2)) # 63.1705%   0.37
print("Dual: ")
print("Accuracy: ", round(ACC_d, 2), "Mean Squared Error: ", round(MSE_d, 2)) # 59.6448%   0.36

# Visualize results in 2 dimensional space.
def visulization(labels, title):
    svd = TruncatedSVD(n_components=2, random_state=42)
    points = svd.fit_transform(x[train_size:])
    plt.scatter(points[:, 0], points[:, 1], label=p_label_p[train_size:])
    plt.title(title)
    plt.show()
visulization(p_label_p[train_size:], "Primal Model Visulization")
visulization(p_label_d[train_size:], "Dual Model Visulization")

# Apply PCA before classification
# using SVD instead due to sparse matrix
svd = TruncatedSVD(n_components=20)
xPCA = svd.fit_transform(x[train_size:])
p_label_p_PCA, p_acc_p_PCA, p_val_p_PCA = predict(y[train_size:], xPCA, m_primal)
p_label_d_PCA, p_acc_d_PCA, p_val_d_PCA = predict(y[train_size:], xPCA, m_dual)
ACC_p_PCA, MSE_p_PCA, SCC_p_PCA = evaluations(y[train_size:], p_label_p_PCA)
ACC_d_PCA, MSE_d_PCA, SCC_d_PCA = evaluations(y[train_size:], p_label_d_PCA)
print("---------------------Using PCA---------------------")
print("Primal: ")
print("Accuracy: ", round(ACC_p_PCA, 2), "Mean Squared Error: ", round(MSE_p_PCA, 2)) # 59.64%  0.4
print("Dual: ")
print("Accuracy: ", round(ACC_d_PCA, 2), "Mean Squared Error: ", round(MSE_d_PCA, 2)) # 59.64%  0.4
