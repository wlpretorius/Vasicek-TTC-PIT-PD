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
from streamlit import caching

st.set_page_config(layout='wide', initial_sidebar_state="expanded")
# sidebarimage = Image.open("C:\\Users\\Admin\\Desktop\\Riskworx\\Riskworx Wordmark Blue.png") 
# st.sidebar.image(sidebarimage, width=250)
df = st.sidebar.file_uploader('Upload your CSV file here:', type='csv')
st.sidebar.header('Navigation')
tabs = ["About","Data Preview and Analysis","6M - LIBOR","6M Fixed Deposit - FCY","6M Fixed Deposit - LCY","Demand Deposits","Savings Deposits","Lending - Foreign","Local Rates","Foreign Deposits"]
page = st.sidebar.radio("Riskworx Pty (Ltd)",tabs)