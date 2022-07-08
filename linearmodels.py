#import modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from tabulate import tabulate
import copy
from scipy import stats

################################################################à



#OLS LINEAR REGRESSION


################################################################à


class ols_lr():
    def __init__(self, data , x, y, cons = True, method = 'non_robust'):
        self.x = x
        self.y = y
        self.data = data
        self.cons = cons
        self.method = method


    #prepare the dataframe

    def prepare_data(self):
        #define list keys for the used df
        keys = [self.y] + self.x
        #select the new df based on keys
        reg_df = self.data[keys].dropna()
        #add the constant if the user wants it
        if self.cons is True:
            ones = np.ones(len(reg_df))
            reg_df['cons']= ones
            #define the features matrix
            X = reg_df.drop(self.y, axis = 1).to_numpy()
            X = X.reshape((len(reg_df), len(self.x)+1))


        else:
            X = reg_df.drop(self.y, axis=1).to_numpy()
            X = X.reshape((len(reg_df), len(self.x)))

        y_vec = reg_df[self.y].to_numpy()
        y_vec = y_vec.reshape((len(y_vec), 1))
        return {'df': reg_df, 'X': X, 'Y': y_vec}

    def betas(self):
        X = self.prepare_data().get('X')
        Y = self.prepare_data().get('Y')

        first_part = np.linalg.inv(np.matmul(np.transpose(X), X))
        second_part = np.matmul(np.transpose(X), Y)
        return np.matmul(first_part, second_part).reshape((1, X.shape[1]))

    def fitted(self):
        X =self.prepare_data().get('X')
        return np.matmul(X,self.betas()[0].reshape((X.shape[1],1)))

    def residuals(self):
        return self.prepare_data().get('Y') - np.matmul(self.prepare_data().get('X'), np.transpose(self.betas()))

    def std(self):
        if self.method == 'non_robust':
            ssr = np.matmul(np.transpose(self.residuals()), self.residuals())
            dfg = 1 / (len(self.prepare_data().get('df')) - self.prepare_data().get('X').shape[1])
            estimated_var = ssr * dfg
            xxinv = np.linalg.inv(np.matmul(np.transpose(self.prepare_data().get('X')), self.prepare_data().get('X')))
            avar = np.multiply(estimated_var, xxinv)
            std = []
            for i in range(avar.shape[1]):
                for j in range(avar.shape[1]):
                    if i == j:
                        std.append(np.sqrt(avar[i, j]))

        elif self.method == 'robust':
            xxinv= np.linalg.inv(np.matmul(np.transpose(self.prepare_data().get('X')), self.prepare_data().get('X')))
            B = np.dot(np.transpose(self.prepare_data().get('X')), self.prepare_data().get('X')* self.residuals()**2)

            avar = np.matmul(np.matmul(xxinv, B), xxinv)
            std = []
            for i in range(avar.shape[1]):
                for j in range(avar.shape[1]):
                    if i == j:
                        std.append(np.sqrt(avar[i, j]))

        return std

    def t(self):
        coeff = self.betas()[0]
        std = self.std()
        t = np.zeros(len(coeff))
        for i in range(len(coeff)):
            t[i] = coeff[i] / (std[i])
        return t
    def p_value(self):
        tvec = self.t()
        pvalues = np.zeros(len(tvec))
        for i in range(len(tvec)):
            pvalues[i] = round(scipy.stats.norm.sf(abs(tvec[i])) * 2, 2)
        return pvalues

    def confidence(self):
        betas = np.array(self.betas()[0])
        std = np.array(self.std())
        low = betas - 1.96*std
        high = betas + 1.96*std

        return {'low': low, 'high': high}

    def summary(self):
        header = [self.y, 'coefficient', 'se', 't', 'p_value', 'low 95', 'high 95']
        table = []
        vars =  self.x +['cons']
        vec = [vars, self.betas()[0], self.std(), self.t(), self.p_value(), self.confidence().get('low'), self.confidence().get('high')]
        vec = list(map(list, zip(*vec)))


        print('Linear regression')
        print('---------------------------------------------------------------')
        print(tabulate(vec, headers=header))
        print('---------------------------------------------------------------')
        return ' '


class two_sls():
    def __init__(self, data , exogenous,endogenous, instruments, y, method = 'non_robust'):
        self.exogenous = exogenous
        self.endogenous = endogenous
        self.instruments = instruments
        self.y = y
        self.data = data

        self.method = method

    def prepare_df_first_stage(self):
        df = self.data
        ones = np.ones(len(df))
        df['cons'] = ones
        first_stage_keys =  self.endogenous + self.exogenous +self.instruments + ['cons']
        df_first_stage = df[first_stage_keys].dropna()
        Z = df_first_stage.drop(self.endogenous,axis = 1).to_numpy()
        xk_vec = []
        for k in self.endogenous:
            xk_vec.append(df_first_stage[k].to_numpy())

        return {'xk': np.transpose(np.array(xk_vec)), 'Z': Z, 'df': df_first_stage}

    def compute_x_hat(self):
        Z = self.prepare_df_first_stage().get('Z')
        Xk = self.prepare_df_first_stage().get('xk')
        df = self.prepare_df_first_stage().get('df')
        z_features = self.exogenous + self.instruments
        X_hat = []
        for i in range(Xk.shape[1]):
            reg = ols_lr(data = df, y = self.endogenous[i], x = z_features)
            fitted = reg.fitted()
            X_hat.append(fitted)

        return np.transpose(np.array(X_hat))[0]


    def prepare_df_second_stage(self):
        X_hat = self.compute_x_hat()
        df_full = self.data
        for i in range(len(self.endogenous)):
            df_full[f'fitted_{self.endogenous[i]}'] = X_hat[:,i]

        fitted_keys = [i for i in df_full.columns.to_list() if i.split('_')[0] == 'fitted']
        df_full['cons'] = np.ones(len(df_full))

        full_keys = ['cons'] + self.exogenous + fitted_keys + [self.y] + self.endogenous
        df_for_reg = df_full[full_keys].dropna()

        xhat_keys =['cons'] + self.exogenous +fitted_keys
        x_keys =   ['cons'] + self.exogenous+ self.endogenous

        X = df_for_reg[x_keys].to_numpy()
        Y = df_for_reg[self.y].to_numpy().reshape(len(df_for_reg), 1)
        X_hat_final=  df_for_reg[xhat_keys].to_numpy().reshape(len(df_for_reg),len(xhat_keys))

        return {'X': X, 'Y': Y, 'X_hat': X_hat_final}

    def betas(self):
            X_hat_final = self.prepare_df_second_stage().get('X_hat')
            X =self.prepare_df_second_stage().get('X')
            Y = self.prepare_df_second_stage().get('Y')

            inve = np.linalg.inv(np.matmul(np.transpose(X_hat_final), X))
            final_part = np.matmul(np.transpose(X_hat_final), Y)
            beta_iv = np.matmul(inve, final_part)
            return beta_iv

    def std(self):
        X = self.prepare_df_second_stage().get('X')
        Y = self.prepare_df_second_stage().get('Y')
        X_hat = self.prepare_df_second_stage().get('X_hat')
        beta = (self.betas())

        residuals = Y - np.matmul(X, beta)
        ssr = np.matmul(np.transpose(residuals), residuals)
        sigma_hat = ssr/ (len(X)- X.shape[1])
        SIGMA = sigma_hat * np.linalg.inv(np.matmul(np.transpose(X_hat), X_hat))
        std_dv = []
        for i in range(SIGMA.shape[0]):
            for j in range(SIGMA.shape[1]):
                if i ==j:
                    std_dv.append(np.sqrt(SIGMA[i,j]))

        return std_dv


    def summary(self):
        header = [self.y, 'coefficient', 'se']
        table = []
        vars =  ['cons' ] + self.exogenous+ self.endogenous
        vec = [vars, self.betas(), self.std()]
        vec = list(map(list, zip(*vec)))

        print('2SlS Regression')
        print('---------------------------------------------------------------')
        print(tabulate(vec, headers=header))
        print('---------------------------------------------------------------')
        print(f'INSTRUMENTED: {self.endogenous}')
        print(f'INSTRUMENTS: {self.exogenous+self.instruments}')

        return ' '



df = pd.read_csv('mroz.csv')

obj = two_sls(data = df, exogenous=['exper', 'expersq'],y = 'lwage', endogenous=['educ'], instruments = ['motheduc', 'fatheduc','huseduc'] )



print(obj.summary())


