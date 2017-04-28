import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, Lasso, LassoLarsCV
import xgboost as xgb

from HousePredict.processData import processData


def rmse_cv(model, X_train, y):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)

#Lasso回归
traindata, trainLabels, testdata = processData()
myalphas = [0.0007, 0.0006, 0.0005, 0.0004]

cv_ridge = [rmse_cv(Lasso(alpha = alpha), traindata, trainLabels).mean()
            for alpha in myalphas]
print("cv_ridge",cv_ridge)
cv_ridge = pd.Series(cv_ridge, index=myalphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
plt.show()

model_lasso = Lasso(alpha=0.0005).fit(traindata, trainLabels)
coef = pd.Series(model_lasso.coef_, index=traindata.columns)
print(coef.sort_values())
imp_coef = pd.concat([coef.sort_values().head(10),coef.sort_values().tail(10)])
imp_coef.plot(kind="barh")
plt.show()


#xgb回归
dtrain = xgb.DMatrix(traindata, label = trainLabels)
dtest = xgb.DMatrix(testdata)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)
model.loc[30:,["test-rmse-mean", "train-rmse-mean"]].plot()
plt.show()

alphas = [500]
cv = [rmse_cv(xgb.XGBRegressor(n_estimators=alpha, max_depth=2, learning_rate=0.1), traindata, trainLabels).mean() for alpha in alphas]
cv = pd.Series(cv, index=alphas)
cv.plot()
plt.show()


model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(traindata, trainLabels)

#将lasso和xgb集成
xgb_preds = np.expm1(model_xgb.predict(testdata))
lasso_preds = np.expm1(model_lasso.predict(testdata))
print(len(xgb_preds))
print(len(lasso_preds))
preds = 0.7*lasso_preds + 0.3*xgb_preds
preds = list(map(lambda x: round(x, 2), preds.tolist()))



outputfile = {"SalePrice":preds,"Id":testdata.index}
outputfile = pd.DataFrame(outputfile)
outputfile.to_csv("xgb.csv", index=False)
