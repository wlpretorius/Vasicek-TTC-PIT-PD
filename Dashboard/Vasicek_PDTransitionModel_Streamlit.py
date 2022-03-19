# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 10:04:39 2022

@author: wlpretorius
@description: Streamlit dashboard for one-factor Vasicek TTC PD to PIT PD transition model
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
import streamlit as st
from PIL import Image
import base64
import os

# Setting up streamlit design and tabs
st.set_page_config(layout='wide', initial_sidebar_state="expanded")
# sidebarimage = Image.open("C:\\Users\\Admin\\Desktop\\Riskworx\\Riskworx Wordmark Blue.png") 
# st.sidebar.image(sidebarimage, width=250)
df = st.sidebar.file_uploader('Upload your CSV file here:', type='csv')
st.sidebar.header('Navigation')
tabs = ["About", "Data Preview and Analysis", "Single Factor Model Optimization", "Transition"]
page = st.sidebar.radio("Vasicek one-factor PD Transition Model", tabs)

if page == "About":
    image = Image.open(r"C:\Users\wlpre\OneDrive\Desktop\Python\Projects\Vasicek_TTC_PIT_PD\Vasicek_TTC_PIT_PD\ttcpdtopitpd.jpg")
    # icon = Image.open("C:\\Users\\Admin\\Desktop\\Riskworx\\RWx & Slogan.png")
    st.image(image, width=700)
    st.header("About")
    st.write("This interface is designed for a quick implementation to transform TTC PDs to PIT PDs using the Vasicek one-factor model.")
    def get_binary_file_downloader_html(bin_file, file_label='File'):
        with open(bin_file, 'rb') as f:
            data = f.read()
        bin_str = base64.b64encode(data).decode()
        href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
        return href
    st.markdown(get_binary_file_downloader_html(r"C:\Users\wlpre\OneDrive\Desktop\Python\Projects\Vasicek_TTC_PIT_PD\Vasicek_TTC_PIT_PD\Yang.pdf", 'Paper on the model'), unsafe_allow_html=True)
    st.header("Requirements")
    st.write("TTC PDs and one-factor macroeconomic variable such as GDP.")         
    st.header("How to use")  
    st.write("Upload the CSV file in the left tab. The CSV file should contain the yearly TTC PDs and macroeconomic values. The transition will update automatically. Note, all plots allow zooming.")
    st.write("")
    st.header("Author and Creator")
    st.markdown(""" **[Willem Pretorius](https://www.linkedin.com/in/wlpretorius/)**""")
    st.markdown(""" **[Contact](mailto:wlpretorius@outlook.com)** """)
    st.markdown(""" **[GitHub](https://github.com//wlpretorius)**""")
    st.write("Created on 19/03/2022")
    st.write("Last updated: **19/03/2022**")
    st.write("")
    st.header("More about Streamlit")                        
    st.markdown("Official documentation of **[Streamlit](https://docs.streamlit.io/en/stable/getting_started.html)**")
    
if page == "Data Preview and Analysis":
    st.title("Data Preview and Analysis")    
    st.subheader("Preview: This tab allows scrolling")        
    if df is not None:
         Mev_TTCPd_data = pd.read_csv(df, usecols=["Year", "F1","TTC PDs"], sep=',', parse_dates = ["Year"])
         df = pd.DataFrame(Mev_TTCPd_data)
         df["F1"] = df["F1"].astype(float)
         df["TTC PDs"] = df["TTC PDs"].astype(float)
         df["Year"] = pd.to_datetime(df["Year"]).dt.date
         st.dataframe(df)
         st.write(df)
         st.subheader("Starting Data Point is on this Date:")
         min_date = df.Year.min()
         st.write(min_date)
         max_date = df.Year.max()
         st.subheader("Latest Data Available is on this Date:")
         st.write(max_date)
         st.subheader("Plot Analysis")
         st.line_chart(df.rename(columns={'Year':'index'}).set_index('index'))
         
if page == "Single Factor Model Optimization":
    st.title("Vasicek Single Factor Model Optimization with Linear Regression")      
    if df is not None:
         Mev_TTCPd_data = pd.read_csv(df, usecols=["Year", "F1","TTC PDs"], sep=',', parse_dates = ["Year"])
         df = pd.DataFrame(Mev_TTCPd_data)
         df["F1"] = df["F1"].astype(float)
         df["TTC PDs"] = df["TTC PDs"].astype(float)
         df["Year"] = pd.to_datetime(df["Year"]).dt.date
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
         st.subheader("Standardising the input data")   
         st.write(standardized_df)
         
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
         fig = plt.figure(figsize=(5,2))
         ax = fig.add_subplot(111)
         ax.set_title('Linear Regression Modelling')
         ax.scatter(X, Y)
         plt.plot(X, std_TTCPd_pred, color='red')
         ax.legend(loc='best')
         st.pyplot(fig)
         
         fig = plt.figure(figsize=(5,2))
         ax = fig.add_subplot(111)
         ax.set_title('TTC Modelling')
         plt.plot(df_regressor['Actual'], label = "Actual")
         plt.plot(df_regressor['Predicted'], label = "Predicted")
         ax.legend(loc='best')
         st.pyplot(fig)
         # Calculating alpha and beta regression parameters
         alpha = linear_regressor.coef_[0]
         c_intercept  = linear_regressor.intercept_[0]
         # Average predicted values and standard deviation
         mean_std_TTCPd_pred = np.mean(std_TTCPd_pred)
         standarddev_std_TTCPd_pred = np.std(std_TTCPd_pred, ddof=1)
         # Forecasted standardised TTC PDs
         std_TTCPd_pred = pd.DataFrame(std_TTCPd_pred, columns=['Std pred PDs'])
         std_TTCPd_pred_std = std_TTCPd_pred['Std pred PDs'].sub(std_TTCPd_pred['Std pred PDs'].mean(0), axis=0).div(std_TTCPd_pred['Std pred PDs'].std(0), axis=0)
         st.subheader("Forecasted standardised TTC PDs")
         st.write(std_TTCPd_pred_std) # Standarised predicted PDs
         
         # Vasicek Fit
         beta = linear_regressor.coef_[0]
         st.subheader("Beta from Linear Regression coefficient:")
         st.write(beta)

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
         
         st.subheader("Optimal Beta minimizing the SSE:")
         st.write(beta)
         st.subheader("The Vasicek Fit:")
         st.write(vasicek_fit)
  
df_NRG = st.sidebar.file_uploader('Upload the PD matrix in CSV format here:', type='csv')         
if page == "Transition":
    st.title("The One-year PIT PD Transition matrix")       
    if df is not None:
         Mev_TTCPd_data = pd.read_csv(df, usecols=["Year", "F1","TTC PDs"], sep=',', parse_dates = ["Year"])
         df = pd.DataFrame(Mev_TTCPd_data)
         df["F1"] = df["F1"].astype(float)
         df["TTC PDs"] = df["TTC PDs"].astype(float)
         df["Year"] = pd.to_datetime(df["Year"]).dt.date
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

         # Calculating alpha and beta regression parameters
         alpha = linear_regressor.coef_[0]
         c_intercept  = linear_regressor.intercept_[0]
         # Average predicted values and standard deviation
         mean_std_TTCPd_pred = np.mean(std_TTCPd_pred)
         standarddev_std_TTCPd_pred = np.std(std_TTCPd_pred, ddof=1)
         # Forecasted standardised TTC PDs
         std_TTCPd_pred = pd.DataFrame(std_TTCPd_pred, columns=['Std pred PDs'])
         std_TTCPd_pred_std = std_TTCPd_pred['Std pred PDs'].sub(std_TTCPd_pred['Std pred PDs'].mean(0), axis=0).div(std_TTCPd_pred['Std pred PDs'].std(0), axis=0)
         
         # Vasicek Fit
         beta = linear_regressor.coef_[0]
         st.subheader("Beta from Linear Regression coefficient:")
         st.write(beta)

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
         
         st.subheader("Optimal Beta minimizing the SSE:")
         st.write(beta)
         
         ############################################################################
         #number1 = st.number_input("Insert Macroeconomic forecast for year 1")
         #number2 = st.number_input("Insert Macroeconomic forecast for year 2")
         #number3 = st.number_input("Insert Macroeconomic forecast for year 3")
         #st.write("The currenct Macroeconomic forecasts are", number1, number2, number3)
         forecast_years = 5
         years_repeat_at = 3
         st.write("Number of years forecasted into the future: ", forecast_years)
         st.write("Number of years repeat at: ", years_repeat_at)

         data = {'Year': ['1', '2', '3'], 'F1_Forecast': [0.03, -0.1, 0.1]}
         F1_Forecast = pd.DataFrame(data)
         F1_Forecast["F1_Forecast"] = F1_Forecast["F1_Forecast"].astype(float)
         st.write(F1_Forecast)

         # Standardising the forecasted values
         F1_Forecast['Std_forecasted_F1'] = (F1_Forecast["F1_Forecast"] - mu_F1) / sigma_F1
         beta_sum_alpha_X = beta * alpha * F1_Forecast['Std_forecasted_F1'][2]
         adjusted_beta = np.sqrt(1 - beta ** 2)

         ############################################################################
         # Transitions - Importing NGR Data
         NGR_Data = pd.read_csv(df_NRG, sep = ',', header = None, index_col = None)
         df_ngr = pd.DataFrame(NGR_Data)

         one_year_TTCPD_trans_matrix = NGR_Data.iloc[0:14:,1:15]
         one_year_TTCPD_trans_matrix = one_year_TTCPD_trans_matrix.append(pd.Series(0, index=one_year_TTCPD_trans_matrix.columns), ignore_index=True)
         one_year_TTCPD_trans_matrix['Def'] = 1 - (one_year_TTCPD_trans_matrix[list(one_year_TTCPD_trans_matrix.columns)].sum(axis=1))
         counter = len(one_year_TTCPD_trans_matrix.columns)
         
         st.subheader("The One-year TTC PD Matrix")
         st.write(one_year_TTCPD_trans_matrix)
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
         
         st.subheader("The One-year PIT PD Matrix")
         st.write(one_year_pit_pd_matrix)

