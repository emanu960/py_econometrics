import linear_regression_model as ols
import pandas as pd

class select_model():

    def linear_regression(self,data , x, y, cons = True, method = 'non_robust'):

        return ols.ols_lr(data, x,y,cons,method)



df = pd.read_csv('body.csv')

print(df)

model = select_model().linear_regression(data = df, y = 'lbwght', x = ['male', 'parity', 'lfaminc', 'packs'])
print(model.betas())