# py_econometrics
econometrics tools on python 


Ols linear regression module

The class ols_lr is used to run the linear regression model. You can create an object by calling this class with the following inputs:
1. data is your dataframe. The dataframe must be organized with rows equals to observations and columns equal to regressors and dependent variable. You don't need to create a column for the constant.
2. x is the list of the strings of your dependent variables you want to use to regress y.
3. y is a string element with the name of the column where the dependent variable is located.
4.method is optional. if you don't call it, you will compute a homoskedastic non robust std. if method == 'robust', you will compute robust standard errors.

After calling the class, you have different functions:

1. betas returns the beta coefficients
2. std returns the vector of standard errors
3. t returns the vector of t statistics
4. p value returns the p value vector
5. summary returns a tabulate version of the model with all data you need.




*******************************************
THIS IS NOT THE FINAL PROJECT. I WILL ADD STEP BY STEP ALL THE FUNCTIONS THAT YOU NEED FORO YOUR ECONOMETRIC WORK.
*******************************************

NEXT STEP: hypothesis tests


#############################################


IF YOU WANT TO USE IT RIGHT NOW, YOU HAVE TO USE LINEARMODELS


#############################################
