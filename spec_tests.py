import linearmodels as lm
import pandas as pd
import numpy as np
from scipy import stats
################################################################à



#TEST FOR ENDOGENEITY


################################################################à


def endo_test(data, exog, endog_to_test,instruments,dependent ):

    if isinstance(endog_to_test,str):
        # first stage regression
        v2_reg = lm.ols(data=data, x=exog + instruments, y=endog_to_test)
        v2_residuals = v2_reg.residuals()
        # prepare df
        df = v2_reg.prepare_data().get('df')
        df[dependent] = data[dependent]
        df = df.drop('cons', axis=1)
        df['v2hat'] = v2_residuals
        # regression test
        reg = lm.ols(data=df, x=exog + [endog_to_test] + ['v2hat'], y=dependent)
        return reg.summary()

    elif isinstance(endog_to_test, list):
        index = np.zeros(len(data))
        for i in range(len(data)):
            index[i] = i
        data['index'] = index
        #first stage regression
        for i in endog_to_test:
            v2_reg = lm.ols(data=data, x=exog + instruments, y=i)
            v2_residuals = v2_reg.residuals()
            df_get = v2_reg.prepare_data().get('df')
            df_get[f'{i}_hat'] = v2_residuals
            index = np.zeros(len(data))
            for i in range(len(data)):
                index[i] = i
            df_get['index'] = index
            data = data.merge(df_get, how = 'inner', on = 'index',suffixes=('', '_right'))

        hat_list = [f'{i}_hat' for i in endog_to_test]
        keys = [dependent]+ exog+instruments+hat_list + endog_to_test
        data = data[keys]
        # regression test
        reg = lm.ols(data=data, x=exog + endog_to_test + hat_list, y=dependent)
        return {'Wald test':lm.Wald_test(reg,hat_list).get('Wald test'), 'p value ':lm.Wald_test(reg,hat_list).get('p_value')[0][2]}



################################################################à



#TEST FOR HETEROSKEDASTICITY


################################################################à


def white_test(data,dependent, regressors):
    obj = lm.ols(data = data, x = regressors, y = dependent)
    fitted = obj.fitted()
    residuals = obj.residuals()
    reisuals_sq = residuals**2
    fitted_sq = fitted**2
    df = obj.prepare_data().get('df')
    df['residuals_sq'] = reisuals_sq
    df['fitted'] = fitted
    df['fitted_sq'] =fitted_sq

    white = lm.ols(data = df, y = 'residuals_sq', x = ['fitted', 'fitted_sq'])
    white_res = white.residuals()
    RSS = np.matmul(np.transpose(white_res), white_res)
    avg_y = np.mean(df[white.y])
    demean = df[white.y ]-avg_y
    SSt = np.matmul(np.transpose(demean), demean)
    R = 1- (RSS/SSt)
    F = (R/2) / ((1-R) / (len(white.prepare_data().get('df')) -3 ))
    p = 1 - (stats.f.cdf(F, 2, (len(white.prepare_data().get('df')) -3 )))

    return {'F': F, 'p_value': p }
