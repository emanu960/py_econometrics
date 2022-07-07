import pandas as pd
import numpy as np
import ols_linear_reg as ols

df = pd.read_csv('body.csv')

print(df)

obj = ols.ols_lr(data = df,y = 'lbwght', x = ['male', 'parity', 'lfaminc', 'packs'])

diagnostic = ols.diagnostic(obj, ['packs'])

print(obj.summary())

print(diagnostic.F_test())


