# PY Econometrics
### Econometrics tools on python 

## Overview

PY Econometrics is a module that allows you to perform different kind of econometric analysis, from standard OLS to most
advanced models.
It is a fresh module and i'm working on it. This mean that you will not find everything right now or you could find some bugs and so on...

By now, you can find all models in *linearmodels.py*. The *test* file is to test all functions. CSV files are dataframes that i use to test and make
a comparison with others statistic programs.

Remember: For now, the "main file" is the *linearmodels.py* file. 


## Models
### OLS linear regression model

###### Inputs
The OLS linear regression model can be called by using the class *ols* in the main file. To call this class, you need the following inputs:

- *data* is the dataframe where you stored your data you want to pass in the ols model.
- *x* is the list of strings of all explanatory variables you want to use to explain y. I mean the strings of the columns that store your x's variables.
**It is very important that you pass a list of strings even if your explanatory variable is just one variable.**
- *y* is the string of the column that stores your dependent variable in your dataframe.
- *cons* is True by default, but if you want to regress without intercept, just declare *cons = False* .
- *method* is "non_robust" by default, but if you want to a Heteroskedastcity robust variance covariance matrix, just declare *method = 'robust'* 

###### Features
- To summarize the results, just call "your object name" . *summary()*. For now, it will print just the table with betas, std, t, p value, and confidence
interval.

- Use *betas()* if you want to obtain beta coefficients.

- Use *fitted()* if you want fitted values.

- Use *residuals()* if you want residuals.

###### Example 

To initialize the model:

```
model  = lm.ols(data = df, x=['exper','expersq','kidslt6','kidsge6'],y = 'lwage')
```



*******************************************
THIS IS NOT THE FINAL PROJECT. I WILL ADD STEP BY STEP ALL THE FUNCTIONS THAT YOU NEED FORO YOUR ECONOMETRIC WORK.
*******************************************

NEXT STEP: hypothesis tests


#############################################


IF YOU WANT TO USE IT RIGHT NOW, YOU HAVE TO USE LINEARMODELS


#############################################
