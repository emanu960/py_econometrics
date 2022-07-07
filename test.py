import pandas as pd
import numpy as np
import ols_linear_reg as ols

df = pd.read_csv('mroz.csv')

print(df)

obj = ols.ols_lr(data = df,y = 'lwage', x = ['exper', 'expersq', 'educ', 'age', 'kidslt6', 'kidsge6'], method ='robust')

print(obj.summary())



diagno = ols.diagnostic(obj,var_to_test=['age', 'kidslt6', 'kidsge6'])

print(diagno.wald_test())