# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 11:48:39 2022

@author: wlpretorius
@description: one-factor Vasicek TTC PD to PIT PD transition model
"""

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
from statistics import NormalDist
from scipy.optimize import minimize

# Data importing
Mev_TTCPd_data = pd.read_csv(r"Data.csv", usecols=["Year", "F1","TTC PDs"], sep=',', parse_dates = ["Year"])
df = pd.DataFrame(Mev_TTCPd_data)
df["F1"] = df["F1"].astype(float)
df["TTC PDs"] = df["TTC PDs"].astype(float)

# Calculating the mean and std of the macroeconomic variable
mu_F1 = df["F1"].mean()
sigma_F1 = df["F1"].std()
      
# Calculating the mean and std of the TTC PDs     
mu_TTCPd = df["TTC PDs"].mean()
sigma_TTCPd = df["TTC PDs"].std()

# Standardizing the macro economic variable and the TTC PD
std_F1 = df["F1"].sub(df["F1"].mean(0), axis=0).div(df["F1"].std(0), axis=0)
std_TTCPd = df["TTC PDs"].sub(df["TTC PDs"].mean(0), axis=0).div(df["TTC PDs"].std(0), axis=0)
standardized_df = {'F1': std_F1, 'TTC PDs': std_TTCPd}
standardized_df = pd.DataFrame(standardized_df)
standardized_df.plot()

# Linear regression for the Vasicek Model to get regressed PDs
# y = alpha(x) + c, alpha = slope, c = intercept
# Regressing standardised macro economic variable on standardised TTCPD
X = std_F1.values.reshape(-1, 1)  # F1 values converted it into a numpy array
Y = std_TTCPd.values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
linear_regressor = LinearRegression() # create object for the class
linear_regressor.fit(X, Y)  # perform linear regression: STD_F1 on STD_TTCPDs
std_TTCPd_pred = linear_regressor.predict(X)  # future predictions 
np.array(std_TTCPd_pred).ravel() # Predicted TTC PD values: y = (alpha)x + c
df_regressor = pd.DataFrame({'Actual': np.array(Y).ravel(), 'Predicted': np.array(std_TTCPd_pred).ravel()})
# Plotting
plt.scatter(X, Y)
plt.plot(X, std_TTCPd_pred, color='red')
df_regressor.plot()
# Calculating alpha and beta regression parameters
alpha = linear_regressor.coef_[0]
c_intercept  = linear_regressor.intercept_[0]
# Average predicted values and standard deviation
mean_std_TTCPd_pred = np.mean(std_TTCPd_pred)
standarddev_std_TTCPd_pred = np.std(std_TTCPd_pred, ddof=1)

# Standardizing TTC PDs again
std_TTCPd_pred = pd.DataFrame(std_TTCPd_pred, columns=['Std pred PDs'])
std_TTCPd_pred.head()
std_TTCPd_pred_std = std_TTCPd_pred['Std pred PDs'].sub(std_TTCPd_pred['Std pred PDs'].mean(0), axis=0).div(std_TTCPd_pred['Std pred PDs'].std(0), axis=0)
std_TTCPd_pred_std # Standarised predicted PDs
plt.scatter(X, std_TTCPd_pred_std)
plt.plot(X, std_TTCPd_pred_std, color='red')
std_TTCPd_pred_std.plot()

# Vasicek Fit
beta = -0.185243572520583 # This is the value from Excel
beta = linear_regressor.coef_[0]

def Vasicek_fit(beta, std_TTCPd_pred_std):
    """
    This function returns the Vasicek Fit using the standardised TTC Pds
    and the mean of the TTC PDs for each year provided. 
    """
    vasicek_fit = []
    ppf = []
    for i in range(0, len(std_TTCPd_pred_std)):
        norm_inv = NormalDist().inv_cdf(mu_TTCPd)
        ppf.append((norm_inv - (beta * std_TTCPd_pred_std[i])) / np.sqrt(1 - beta ** 2))
        vasicek_fit.append(norm.cdf(ppf[i]))
    
    return np.array(vasicek_fit).ravel()


Vasicek_fit(beta = linear_regressor.coef_[0], std_TTCPd_pred_std = std_TTCPd_pred_std)
vasicek_fit = Vasicek_fit(beta= -0.185243572520583, std_TTCPd_pred_std = std_TTCPd_pred_std)

original_TTC_PDs = df["TTC PDs"].values.flatten()

############################################################################
# The objective function to minimize (y = sum of squares error) by changing Beta
def f(b): 
     y = np.sum((Vasicek_fit(b, std_TTCPd_pred_std) - original_TTC_PDs) ** 2)
     return y

# Starting guess
x_start = 0

# Optimizing
result = minimize(f, x_start, method = 'BFGS')
beta = result.x

vasicek_fit = Vasicek_fit(beta = result.x, std_TTCPd_pred_std = std_TTCPd_pred_std)

############################################################################
data = {'Year': ['1', '2', '3'], 'F1_Forecast': [0.03, -0.1, 0.1]}
F1_Forecast = pd.DataFrame(data)
F1_Forecast["F1_Forecast"] = F1_Forecast["F1_Forecast"].astype(float)

# Standardising the forecasted values
F1_Forecast['Std_forecasted_F1'] = (F1_Forecast["F1_Forecast"] - mu_F1) / sigma_F1
beta_sum_alpha_X = beta * alpha * F1_Forecast['Std_forecasted_F1'][2]
adjusted_beta = np.sqrt(1 - beta ** 2)

############################################################################
# Transitions - Importing NGR Data
NGR_Data = pd.read_csv(r"NGR Data.csv", sep = ',', header = None, index_col = None)
df_ngr = pd.DataFrame(NGR_Data)

one_year_TTCPD_trans_matrix = NGR_Data.iloc[0:14:,1:15]
one_year_TTCPD_trans_matrix = one_year_TTCPD_trans_matrix.append(pd.Series(0, index=one_year_TTCPD_trans_matrix.columns), ignore_index=True)
one_year_TTCPD_trans_matrix['Def'] = 1 - (one_year_TTCPD_trans_matrix[list(one_year_TTCPD_trans_matrix.columns)].sum(axis=1))
counter = len(one_year_TTCPD_trans_matrix.columns)
forecast_years = 5
years_repeat_at = 3

dummy_matrix = pd.DataFrame(0, index=np.arange(len(one_year_TTCPD_trans_matrix)), columns=one_year_TTCPD_trans_matrix.columns)

for i in range(0,dummy_matrix.shape[0]):
    for j in range(0,dummy_matrix.shape[1]):
        dummy_matrix.iloc[j,i] = one_year_TTCPD_trans_matrix.iloc[j,i:].sum()
        
        
new_dummy_matrix = dummy_matrix.copy()
new_dummy_matrix = new_dummy_matrix.drop(1,1)
new_dummy_matrix.columns = dummy_matrix.columns[:-1] 
new_dummy_matrix.loc[:,"Def"] = dummy_matrix["Def"]

################################################################################################
# one_year_PiTPD_trans_matrix
# Populating the PiT PD Matrix
matrix = []
for i in range(0,dummy_matrix.shape[0]):
    row_values = []
    for j in range(0,dummy_matrix.shape[1]):
        transition_list_dummy = []
        transition_list_new_dummy = []
        if (dummy_matrix.iloc[i,j] > 0.99999):
            transition_list_dummy.append(NormalDist().inv_cdf(0.99999))
        else:
            transition_list_dummy.append(NormalDist().inv_cdf(dummy_matrix.iloc[i,j]))
            
        norm.cdf((transition_list_dummy - beta_sum_alpha_X) / adjusted_beta)

        if (new_dummy_matrix.iloc[i,j] > 0.99999):
            transition_list_new_dummy.append(NormalDist().inv_cdf(0.99999))
        else:
            transition_list_new_dummy.append(NormalDist().inv_cdf(new_dummy_matrix.iloc[i,j]))
            
        norm.cdf((transition_list_new_dummy - beta_sum_alpha_X) / adjusted_beta)
        values = norm.cdf((transition_list_dummy - beta_sum_alpha_X) / adjusted_beta) - norm.cdf((transition_list_new_dummy - beta_sum_alpha_X) / adjusted_beta)  
        row_values.append(values[0])
    matrix.append(row_values)

one_year_pit_pd_matrix = pd.DataFrame(matrix, columns=one_year_TTCPD_trans_matrix.columns)
one_year_pit_pd_matrix.loc[:,"Def"] = dummy_matrix["Def"]

# Calculating the Defaults
for i in range(0, dummy_matrix.shape[0] - 1):
    one_year_pit_pd_matrix.iloc[i,14] = norm.cdf((NormalDist().inv_cdf(dummy_matrix["Def"][i]) - beta_sum_alpha_X) / adjusted_beta)

#####################################################################################################
df1 = one_year_pit_pd_matrix.to_numpy()
df2 = one_year_pit_pd_matrix.to_numpy()
df3 = one_year_pit_pd_matrix.to_numpy()

df_mult1 = np.matmul(df1, df2)
df_mult2 = np.matmul(df_mult1, df3)

df_dot1 = np.matmul(df1, df2)
df_dot2 = np.matmul(df_dot1, df3)
