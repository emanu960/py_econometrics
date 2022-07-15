import pandas as pd
import numpy as np
import linearmodels as lm

df = pd.read_csv('mroz.csv')

model = lm.two_sls(data = df, exogenous=['exper','expersq','kidslt6','kidsge6'],y = 'lwage', endogenous=['educ'], instruments = ['motheduc','fatheduc','huseduc'], method = 'robust')

print(model.summary())

print(lm.Wald_test(model, var_to_test=['kidslt6','kidsge6']))



# df = pd.read_csv('card.csv')
#
# df['blackeduc'] = df['black'] * df['educ']
# df['blacknearc'] = df['black'] * df['nearc4']
#
# print(endo_test(df, ['exper', 'expersq','black','south', 'smsa', 'reg661', 'reg662', 'reg663','reg664','reg665', 'reg666', 'reg667', 'reg668', 'smsa66'],
#                 ['educ','blackeduc'], ['nearc4','blacknearc' ], 'lwage'))
