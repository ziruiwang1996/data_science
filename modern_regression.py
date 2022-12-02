# Least Squares Regression, Ridge Regression, and LASSO to predict the target variable.
# Train on blogData Train.csv and test on blogData test-2012.03.31.01 00.csv.
# Report RMSE for each model.

import numpy as np
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# import data
train = np.genfromtxt("BlogFeedback/blogData_train.csv", delimiter="," )
test = np.genfromtxt("BlogFeedback/blogData_test-2012.03.31.01_00.csv", delimiter="," )

# cross validation
kf = KFold(n_splits=10)
def cross_validation():
    ridge_cro_val_result = dict()
    lasso_cro_val_result = dict()

    for alp in range(1, 20, 2):
        ridge_rmsd_by_fold = []
        lasso_rmsd_by_fold = []
        for tra, val in kf.split(train):
            # Ridge Regression
            ridge_reg = linear_model.Ridge(alpha=alp)
            # LASSO
            lasso_reg = linear_model.Lasso(alpha=alp)

            # dimension 0-280 are features, and d-281 is the label
            ridge_reg.fit(train[tra, :280], train[tra, -1])
            lasso_reg.fit(train[tra, :280], train[tra, -1])

            pred_ridge = ridge_reg.predict(train[val, :280])
            pred_lasso = lasso_reg.predict(train[val, :280])

            ridge_rmsd_by_fold.append(mean_squared_error(y_true=train[val, -1], y_pred=pred_ridge, squared=False))
            lasso_rmsd_by_fold.append(mean_squared_error(y_true=train[val, -1], y_pred=pred_lasso, squared=False))

        ridge_cro_val_result[alp] = sum(ridge_rmsd_by_fold)/len(ridge_rmsd_by_fold)
        lasso_cro_val_result[alp] = sum(lasso_rmsd_by_fold)/len(lasso_rmsd_by_fold)
    print('ridge', ridge_cro_val_result)
    print('lasso', lasso_cro_val_result)

#ridge {1: 26.28954387199723, 3: 26.263413175518952, 5: 26.2493699164313, 7: 26.240067653571096, 9: 26.233204760001666, 11: 26.227803691798954,
#      13: 26.223367705044712, 15: 26.21961264931977, 17: 26.21636178589274, 19: 26.213498137678563}
#lasso {1: 25.963108848289473, 3: 26.396449621808717, 5: 26.658154419288536, 7: 26.65012995533477, 9: 26.721416394606955, 11: 26.79231452964415,
#      13: 26.86824701077801, 15: 26.947483403220264, 17: 27.02443839640386, 19: 27.08627863541871}

# Least Squares Regression
reg_ls = linear_model.LinearRegression()
# Ridge Regression
ridge_reg = linear_model.Ridge(alpha=18)
# LASSO
lasso_reg = linear_model.Lasso(alpha=1)

# training
reg_ls.fit(train[:, :280], train[:, -1])
ridge_reg.fit(train[:, :280], train[:, -1])
lasso_reg.fit(train[:, :280], train[:, -1])

# prediction
pred_ls = reg_ls.predict(test[:, :280])
pred_ridge = ridge_reg.predict(test[:, :280])
pred_lasso = lasso_reg.predict(test[:, :280])

# root mean squared error
rmse_ls = mean_squared_error(y_true=test[:, -1], y_pred=pred_ls, squared=False)
rmse_ridge = mean_squared_error(y_true=test[:, -1], y_pred=pred_ridge, squared=False)
rmse_lasso = mean_squared_error(y_true=test[:, -1], y_pred=pred_lasso, squared=False)
print("Least Squares Regression RMSE: ", round(rmse_ls, 2))
print("Ridge Regression RMSE: ", round(rmse_ridge, 2))
print("LASSO Regression RMSE: ", round(rmse_lasso, 2))
#Least Squares Regression RMSE:  40.4
#Ridge Regression RMSE:  40.44
#LASSO Regression RMSE:  40.84

## What are the most important features according to LASSO?
# Answer: important features are domensions where have non-zero coefficient
print(lasso_reg.coef_)
"""[ 8.44069887e-02  1.50551502e-01 -0.00000000e+00 -6.29296353e-04
  9.32923600e-02  6.90030911e-01  0.00000000e+00 -0.00000000e+00
  1.09449782e-03  1.75521135e-01  0.00000000e+00 -0.00000000e+00
  0.00000000e+00 -1.94932444e-02  0.00000000e+00 -4.15031916e-01
 -5.95777013e-02 -0.00000000e+00  1.10404464e-03  2.68489923e-02
  0.00000000e+00 -0.00000000e+00 -1.18768486e-02  3.71928058e-03
  9.74284679e-02 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
  0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
  0.00000000e+00  0.00000000e+00 -0.00000000e+00 -0.00000000e+00
 -0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00  0.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
  0.00000000e+00  0.00000000e+00 -3.93692482e-02  1.82535707e-01
 -0.00000000e+00 -3.02322175e-02  3.88990600e-02  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
 -1.49445515e-01  1.78245802e-04  0.00000000e+00 -0.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00  0.00000000e+00
  0.00000000e+00 -0.00000000e+00 -0.00000000e+00  0.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
  0.00000000e+00 -0.00000000e+00  0.00000000e+00 -0.00000000e+00
 -0.00000000e+00 -0.00000000e+00  0.00000000e+00 -0.00000000e+00
  0.00000000e+00 -0.00000000e+00 -0.00000000e+00  0.00000000e+00
 -0.00000000e+00  0.00000000e+00 -0.00000000e+00  0.00000000e+00
  0.00000000e+00 -0.00000000e+00 -0.00000000e+00  0.00000000e+00
 -0.00000000e+00  0.00000000e+00 -0.00000000e+00 -0.00000000e+00
 -0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
 -0.00000000e+00  0.00000000e+00 -0.00000000e+00 -0.00000000e+00
  0.00000000e+00 -0.00000000e+00 -0.00000000e+00  0.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
  0.00000000e+00  0.00000000e+00 -0.00000000e+00  0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00 -0.00000000e+00
  0.00000000e+00  0.00000000e+00 -0.00000000e+00  0.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
 -0.00000000e+00 -0.00000000e+00  0.00000000e+00 -0.00000000e+00
 -0.00000000e+00  0.00000000e+00 -0.00000000e+00 -0.00000000e+00
 -0.00000000e+00  0.00000000e+00  0.00000000e+00 -0.00000000e+00
 -0.00000000e+00 -0.00000000e+00  0.00000000e+00 -0.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
 -0.00000000e+00 -0.00000000e+00  0.00000000e+00 -0.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
 -0.00000000e+00  0.00000000e+00 -0.00000000e+00  0.00000000e+00
 -0.00000000e+00  0.00000000e+00 -0.00000000e+00 -0.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00  0.00000000e+00
  0.00000000e+00 -0.00000000e+00  0.00000000e+00 -0.00000000e+00
 -0.00000000e+00 -0.00000000e+00  0.00000000e+00 -0.00000000e+00
 -0.00000000e+00  0.00000000e+00 -0.00000000e+00  0.00000000e+00
  0.00000000e+00 -0.00000000e+00  0.00000000e+00  0.00000000e+00
 -0.00000000e+00  0.00000000e+00 -0.00000000e+00 -0.00000000e+00
  0.00000000e+00  0.00000000e+00 -0.00000000e+00  0.00000000e+00
 -0.00000000e+00  0.00000000e+00 -0.00000000e+00 -0.00000000e+00
  0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
 -0.00000000e+00  0.00000000e+00  0.00000000e+00 -0.00000000e+00
 -0.00000000e+00  0.00000000e+00 -0.00000000e+00  0.00000000e+00
  0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
  0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
  0.00000000e+00 -0.00000000e+00 -0.00000000e+00  0.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00  0.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00
  0.00000000e+00 -0.00000000e+00  0.00000000e+00 -0.00000000e+00
 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00
  0.00000000e+00 -0.00000000e+00  0.00000000e+00 -0.00000000e+00
 -0.00000000e+00  0.00000000e+00  0.00000000e+00 -0.00000000e+00
  0.00000000e+00  0.00000000e+00  0.00000000e+00  0.00000000e+00]"""
