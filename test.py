import pandas as pd
import numpy as np
import linearmodels as lm

df = pd.read_csv('mroz.csv')

model = lm.two_sls(data = df, exogenous=['exper','expersq','kidslt6','kidsge6'],y = 'lwage', endogenous=['educ'], instruments = ['motheduc','fatheduc','huseduc'], method = 'robust')

print(model.summary())

print(lm.Wald_test(model, var_to_test=['kidslt6','kidsge6']))