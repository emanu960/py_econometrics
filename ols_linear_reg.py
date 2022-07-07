#import modules

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from tabulate import tabulate


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




        print('---------------------------------------------------------------')
        print(tabulate(vec, headers=header))
        print('---------------------------------------------------------------')



class diagnostic(ols_lr):
    def __init__(self, class_ols_sr, var_to_test):
        self.data = class_ols_sr.data
        self.x  = class_ols_sr.x
        self.y = class_ols_sr.y
        self.var_to_test = var_to_test
        self.method = class_ols_sr.method

    def F_statistic(self):
        restricted_var = [i for i in self.x if i not in self.var_to_test]
        restricted = ols_lr(data =self.data , x = restricted_var, y = self.y)
        residuals1 = restricted.residuals()
        ssr1 = np.matmul(np.transpose(residuals1), residuals1)
        y1_mean = np.mean(restricted.prepare_data().get('Y'))
        unrestricted = ols_lr(data =self.data , x = self.x, y = self.y)
        residuals2= unrestricted.residuals()
        y2_mean = np.mean(unrestricted.prepare_data().get('Y'))
        ssr2 = np.matmul(np.transpose(residuals2), residuals2)
        tss2 = np.sum((unrestricted.prepare_data().get('Y') - y2_mean)**2)
        tss1 = np.sum((restricted.prepare_data().get('Y') - y1_mean) ** 2)
        r1 = 1-(ssr1/tss1)
        r2 = 1 - (ssr2 / tss2)
        n = len(unrestricted.prepare_data().get('Y'))
        k2 = unrestricted.prepare_data().get('X').shape[1]
        k1 =restricted.prepare_data().get('X').shape[1]-1
        one = (r2-r1)/(1-r2)
        two = (n-k2)/k1
        return {'F':one * two,'p_value':  1-scipy.stats.f.cdf(one * two, k1, n-k2)} #find p-value of F test statistic  }
