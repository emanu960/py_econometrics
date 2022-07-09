import pandas as pd
import numpy as np
import linearmodels as lm

df = pd.read_csv('mroz.csv')

model = lm.two_sls(data = df, exogenous=['exper','expersq','kidslt6','kidsge6'],y = 'lwage', endogenous=['educ'], instruments = ['motheduc','fatheduc','huseduc'] )

print(lm.Wald_test(model, ['kidslt6','kidsge6']))