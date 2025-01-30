# An approach for good modelling and forecasting of sea surface salinity in a coastal zone using machine learning LASSO regression models built with sparse satellite time series datasets
Written by Opeyemi Ajibola-James, PhD, Geo Inheritance Limited.
License: This work is licensed under Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International. https://creativecommons.org/licenses/by-nc-nd/4.0/

# LIST OF APPENDICES
# Appendix 1: Data preparation (Data Extraction, Transformation and Cleaning Algorithms) 
#Python 3.10.2 scripts with appropriate libraries were executed in Spyder IDE (Integrated #Development Environment) 5.2.2 software to achieve the tasks in this appendix.
#The lines of codes in this particular appendix are relevant to subsection 
#3.2 Data preparation

#(a) Data Extraction from .nc to .csv Files
#The glob module was used to detect all the netCD files inside the working folder and the dataset was imported into the IDE using netCD4 library:
import glob
from netCDF4 import Dataset 

#Pandas was imported as pd in order to create a date range:
import pandas as pd

#Numpy was imported as np to be able extract data from the netCDF files:
import numpy as np 

#Empty list of all the .nc files was created, detected in the directory and printed in the console:
for file in glob.glob('*.nc')
    print(file)
 
#The variable called 'data' was created to see a chronological list of the netCD4 files:
    data = Dataset (file, 'r')
  
#The xarray library was imported as xr to open the dataset and transform its dataframe to .csv:
    import xarray as xr
    data = xr.open_dataset(file)
    df = data.to_dataframe().reset_index()
    df.to_csv('sss_raw_m_mv_data_smap.csv')
 
#(b) Data Transformation 
#This was characterized by deleting all records (rows) with any missing/empty data in the .csv files by running the following lines of code.

#The csv library was imported:
import csv

#The Pandas library was imported as pd
import pandas as pd

#The Pandas was utilized to read the .csv file to generate a dataframe:
df = pd.read_csv('sss_raw_m_mv_data_smap.csv')

#Cells with empty records were dropped in the dataframe:
new_df = df.dropna()

#The dataframe was converted to a .csv file:
new_df.to_csv('sss_ref0001_m_mv_data_smap.csv', index=False)
 
#(c) Data Cleaning via Extraction of Appropriate Data Points for Specific Locations 

#The Pandas library was imported as pd:  
import pandas as pd

#The xarray library was imported as xr:   
import xarray as xr

#The csv library was imported as csv:
import csv

#The numpy library was imported np:
import numpy as np

#The column names were defined:
colnames = ['lat', 'lon', 'lat_lon', 'sss', 'c_lon', 'constant', 'anc_sss', 'anc_sst', 'land_fraction', 'hwind_spd', 'wind_spd', 'sss_uncertainty']

#The Pandas was used to read the .csv files:
data = pd.read_csv('sss_ref0003_apr2015_apr2022_m_mv_data_smap.csv', names = colnames, skiprows=(0,))

lat = data['lat']
lon = data['lon']
lat_lon = ['lat_lon']
sss = data['sss']
c_lon = data['c_lon']
constant = data['constant']
anc_sss = data['anc_sss']
anc_sst  = data['anc_sst']
land_fraction = data['land_fraction']
hwind_spd = data['hwind_spd']
wind_spd = data['wind_spd']	
sss_uncertainty = data['sss_uncertainty']

#The lat_lon values in bold were utilized to determine the locations of the variables extracted for the cEEZ:
df_lat_lon = data[data['lat_lon'].isin([
5.6252875, 5.6253125, 5.6253375, 5.6253625, 5.6253875, 5.6254125, 5.6254375, # 5.6254625, 5.6254875
5.3752875, 5.3753125, 5.3753375, 5.3753625, 5.3753875, 5.3754125, 5.3754375, 5.3754625, # 5.3754875, 5.3755125
5.1252875, 5.1253125, 5.1253375, 5.1253625, 5.1253875, 5.1254125, 5.1254375, 5.1254625, # 5.1254875, 5.1255125
4.8752875, 4.8753125, 4.8753375, 4.8753625, 4.8753875, 4.8754125, 4.8754375, 4.8754625, # 4.8754875, 4.8755125
4.6252875, 4.6253125, 4.6253375, 4.6253625, 4.6253875, 4.6254125, 4.6254375, 4.6254625, 4.6254875, # 4.6255125, 4.6255375
4.3752875, 4.3753125, 4.3753375, 4.3753625, 4.3753875, 4.3754125, 4.3754375, 4.3754625, 4.3754875, 4.3755125, #4.3755375, 4.3755625, 4.3757125, 4.3757375, 4.3757625, 4.3757875, 4.3758125, 4.3758375, 4.3758625
4.1252875, 4.1253125, 4.1253375, 4.1253625, 4.1253875, 4.1254125, 4.1254375, 4.1254625, 4.1254875, 4.1255125, 4.1255375, # 4.1255625, 4.1255875, 4.1256125, 4.1256375, 4.1256625, 4.1256875, 4.1257125, 4.1257375, 4.1257625, 4.1257875, 4.1258125, 4.1258375, 4.1258625
3.8752875, 3.8753125, 3.8753375, 3.8753625, 3.8753875, 3.8754125, 3.8754375, 3.8754625, 3.8754875, 3.8755125, 3.8755375, 3.8755625, #3.8755875, 3.8756125, 3.8756375, 3.8756625, 3.8756875, 3.8757125, 3.8757375, 3.8757625, 3.8757875, 3.8758125, 3.8758375, 3.8758625
3.6252875, 3.6253125, 3.6253375, 3.6253625, 3.6253875, 3.6254125, 3.6254375, 3.6254625, 3.6254875, 3.6255125, 3.6255375, 3.6255625, 3.6255875, 3.6256125, 3.6256375, 3.6256625, 3.6256875, 3.6257125, 3.6257375, 3.6257625, 3.6257875, 3.6258125, # 3.6258375,  
3.3752875, 3.3753125, 3.3753375, 3.3753625, 3.3753875, 3.3754125, 3.3754375, 3.3754625, 3.3754875, 3.3755125, 3.3755375, 3.3755625, 3.3755875, 3.3756125, 3.3756375, 3.3756625, 3.3756875, 3.3757125, 3.3757375, 3.3757625, 3.3757875, 3.3758125, # 3.3758375,  
3.1252875, 3.1253125, 3.1253375, 3.1253625, 3.1253875, 3.1254125, 3.1254375, 3.1254625, 3.1254875, 3.1255125, 3.1255375, 3.1255625, 3.1255875, 3.1256125, 3.1256375, 3.1256625, 3.1256875, 3.1257125, 3.1257375, 3.1257625, 3.1257875, 3.1258125, # 3.1258375, 3.1258625,
2.8752875, 2.8753125, 2.8753375, 2.8753625, 2.8753875, 2.8754125, 2.8754375, 2.8754625, 2.8754875, 2.8755125, 2.8755375, 2.8755625, 2.8755875, 2.8756125, 2.8756375, 2.8756625, 2.8756875, 2.8757125, 2.8757375, 2.8757625, 2.8757875, 2.8758125, # 2.8758375, 2.8758625,
2.6252875, 2.6253125, 2.6253375, 2.6253625, 2.6253875, 2.6254125, 2.6254375, 2.6254625, 2.6254875, 2.6255125, 2.6255375, 2.6255625, 2.6255875, 2.6256125, 2.6256375, 2.6256625, 2.6256875, 2.6257125, 2.6257375, 2.6257625, 2.6257875, 2.6258125, 2.6258375, 2.6258625,
2.3752875, 2.3753125, 2.3753375, 2.3753625, 2.3753875, 2.3754125, 2.3754375, 2.3754625, 2.3754875, 2.3755125, 2.3755375, 2.3755625, 2.3755875, 2.3756125, 2.3756375, 2.3756625, 2.3756875, 2.3757125, 2.3757375, 2.3757625, 2.3757875, 2.3758125, 2.3758375, 2.3758625,
2.1252875, 2.1253125, 2.1253375, 2.1253625, 2.1253875, 2.1254125, 2.1254375, 2.1254625, 2.1254875, 2.1255125, 2.1255375, 2.1255625, 2.1255875, 2.1256125, 2.1256375, 2.1256625, 2.1256875, 2.1257125, 2.1257375, 2.1257625, 2.1257875, 2.1258125, 2.1258375, 2.1258625,
1.8752875, 1.8753125, 1.8753375, 1.8753625, 1.8753875, 1.8754125, 1.8754375, 1.8754625, 1.8754875, 1.8755125, 1.8755375, 1.8755625, 1.8755875, 1.8756125, 1.8756375, 1.8756625, 1.8756875, 1.8757125, 1.8757375, 1.8757625, 1.8757875, 1.8758125, 1.8758375, 1.8758625,
1.6252875, 1.6253125, 1.6253375, 1.6253625, 1.6253875, 1.6254125, 1.6254375, 1.6254625, 1.6254875, 1.6255125, 1.6255375, 1.6255625, 1.6255875, 1.6256125, 1.6256375, 1.6256625, 1.6256875, 1.6257125, 1.625 7375, 1.6257625, 1.6257875, 1.6258125, 1.6258375, 1.6258625])]  

#The lat_lon values in bold were utilized to determine the locations of the variables extracted for the iMA:
df_lat_lon = data[data['lat_lon'].isin([

5.8754375, 
5.6254625, 
5.3754875, 
5.1254875,
4.8754875, 
4.6255125, 
4.3755375,                                              
4.1255625, 4.1257875, 4.1258125, 
3.8755875, 3.8756125, 3.8756375, 3.8756625, 3.8756875, 3.8757125, 3.8757375, 3.8757625])]

#The output dataframe was created and exported to .csv:
df_lat_lon_output=pd.DataFrame(df_lat_lon)
df_lat_lon_output.to_csv('sss_StudyArea_ref0003_Analysed_m_mv_data_smap.csv')

 
# Appendix 2: Least Absolute Shrinkage and Selection Operator (LASSO) Regression Model and Algorithm
#The forecastML package (library) 0.9.1 and other relevant libraries in the R 4.1.3/ R-studio #2022.02.3-492 were imported to achieve the tasks in this appendix.
#The lines of codes in this particular appendix are relevant to subsections 
#3.3 Parameterization
#3.4 Least absolute shrinkage and selection operator regression models and algorithm 
#3.7 Experimental validation of PPVs importance and collinearity with ML LASSO models 
#3.8 Prediction of SSS with the ML LASSO model 

#1. DATA LOADING
#The relevant libraries were imported.
library(glmnet)
library(forecastML)

#The data file (.csv) with 7 variables sss, sst, ws, hws, precip, sla, & adt was uploaded interactively.
data_train_sss_7v <- read.csv(file.choose(), header = TRUE, stringsAsFactors = FALSE)

#For the Parameterization in 3.3, 7 variables of interest were utilized and ordered as appropriate 
data_train_sss <- data_train_sss_7v[, c("sss", "ws", "hws", "sst", "adt", "sla", "precip")]

#For the Experimental validation of PPVs importance and collinearity in 3.7, however, the above 7 #variables of interest were replaced and ordered as appropriate, and run #one after the other to achieve the experiments A-G. Consequently, the below values were edited as appropriate using the #values provided in Table 3 of the manuscript/ #article. For examples, when running the code for the #experiment A, use
##data_train_sss <- data_train_sss_7v[, c("sss", "ws", "hws", "sst", "adt", "sla", "precip")]
#For the experiment D, use
##data_train_sss <- data_train_sss_7v[, c("sss", "ws", "hws", "sla")]  

#The sss variable was set to 5 decimal places.
DT::datatable(head(data_train_sss, 5)) 

#2. BUILDING THE LASSO REGRESSION MODEL
#The Lookback and Horizone were defined and store using appropriate values
#For the Parameterization in 3.3, each of the 6 possible PCs of LB and H values LB:36, H:36; #LB:36, H:24; LB:36, H:12; LB:24, H:24; LB:24, H:12 and LB:12, H:12 was used one after the #other to replace the values below as appropriate.

lookback <- 1:36
horizons <- 1:36 

#For the Experimental validation of PPVs importance and collinearity in 3.7, however, the appropriate PCs values, LB:24, H:12 were utilized to run the experiments A-G one #after the other. 

#The appropriate set seed value was utilized to support replicable iteration of the subsequent lines of codes
set.seed(555)

#The dataset of lagged features for the modeling was created.
data_train <- forecastML::create_lagged_df(data_train_sss, type = "train", method = "direct",
                                           outcome_col = 1, lookback = lookback, horizons = horizons)

#Validation dataset for outer-loop nested cross-validation was created.
windows <- forecastML::create_windows(data_train, window_length = 0)

#The User-define LASSO model was built 
model_fn <- function(data) {
  x <- as.matrix(data[, -1, drop = FALSE])
  y <- as.matrix(data[, 1, drop = FALSE])
    model <- glmnet::cv.glmnet(x, y)
}

#The model was trained across forecast horizons and validation datasets.
model_results <- forecastML::train_model(data_train, windows, model_name = "LASSO", model_function = model_fn)

#The User-defined LASSO prediction function was built
predict_fn <- function(model, data) {
  data_pred <- as.data.frame(predict(model, as.matrix(data)))
}

#SSS prediction on the validation dataset was done using the data_train.
data_fit <- predict(model_results, prediction_function = list(predict_fn), data = data_train)

#The residuals were computed using the residuals’ function.
residuals <- residuals(data_fit)

#The forward-looking forecast data.frame was built.
data_forecast <- forecastML::create_lagged_df(data_train_sss, type = "forecast", method = "direct",
                                              outcome_col = 1, lookback = lookback, horizons = horizons)

#SSS prediction on the validation dataset was finalized using the data_forecast.
data_forecasts <- predict(model_results, prediction_function = list(predict_fn), data = data_forecast)

#The Forward-looking combine forecasts data.frame was built and printed into a .csv file.
data_forecasts <- forecastML::combine_forecasts(data_forecasts)
write.csv(data_forecasts, 'C:\\Users\\A-JAMES\\Documents\\Final_Data_Analysis_2012-2021\\obj_VI_StudyArea_2016_2021_smap_cmem\\LASSO_3pvar_LB12-36_H12-36_final\\data_forecasts_3v_LB24_H12_2021_via_2016_2020.csv')
 
#3. APPLYING THE LASSO REGRESSION MODEL
#The appropriate set seed value was utilized to support replicable iteration of the subsequent lines of codes
set.seed(777)

#The data_forecasts intervals and the level of confidence at 5% (0.05) were computed using the residuals for performance metric
data_forecasts <- forecastML::calculate_intervals(data_forecasts, residuals, 
                                                  levels = seq(.5, .95, .05), times = 200) # levels = seq(0.05, 0.95), times = 200)

#The forecasts plot was generated for each validation dataset.
plot (data_forecasts, data_train_sss[(1:60), ], (1:nrow(data_train_sss))[(1:60)],interval_alpha = seq(.1, .2, length.out = 10))

#The LASSO residual error metrics (1-R2) by Horizon (for each of the steps Ahead forecast) and #error global were computed in terms of MAE, MAPE, MDAPE, SMAPE and RMSE.
return_error(data_fit)

#The residual error metrics were written to a .csv file.
lasso_error <- return_error(data_fit)
write.csv(lasso_error, 'C:\\Users\\A-JAMES\\Documents\\Final_Data_Analysis_2012-2021\\obj_VI_StudyArea_2016_2021_smap_cmem\\LASSO_3pvar_LB12-36_H12-36_final\\2ndlasso_error_Best3v_LB36_H36_2016_2020.csv')

#4. COMPUTING THE ERROR METRICS FOR THE LASSO REGRESSION MODEL
#The the observed and predicated values were stored as appropriate using
sss_obs1 <- data_fit$sss
sss_pred1 <- data_fit$sss_pred

#The R-squared (R2) value for the fitted model was computed
rss <- sum((sss_pred1 - sss_obs1) ^ 2)
tss <- sum((sss_obs1 - mean(sss_obs1)) ^ 2)
rsq <- 1 - rss/tss
rsq

#The MLmetrics library was imported for computing the RMSE.
library(MLmetrics)

#The RMSE was computed.
RMSE(sss_pred1,sss_obs1)
#where sss_pred1 is the predicted SSS and sss_obs1 is the actual satellite SSS (Jan.-Dec. 2021).


# Appendix 3: L0-Regularized Regression Model and Algorithm
#The relevant libraries in the R 4.1.3/ R-studio 2022.02.3-492 were imported to achieve the tasks in #this appendix.
#The lines of codes in this particular appendix are relevant to subsection
#3.6 Determination of PPVs importance and collinearity with L0 models and algorithms

#1. DATA LOADING
#The applicable library was imported.
library(L0Learn)

#The data file (.csv) with 7 variables sss, sst, ws, hws, precip, sla, & adt was uploaded interactively.
data <- read.csv(file.choose(), header = T, stringsAsFactors = TRUE)

#The 6 predictor variables were ordered in the view as appropriate.
xdata <- data[, c("ws", "hws", "sst", "adt", "sla", "precip")]

#The 6 predictor variable were set as matrix, while the target variable was defined as appropriate.
X <- as.matrix(xdata[,-7])
y = data$sss

#2. BUILDING THE L0-REGULARIZED REGRESSION MODEL
#Set seed as appropriate for replication purposes
set.seed(111)

#The L0-regularized regression model was fitted with L0L2 Penalty and CDPSI Algorithm.
L0L2_CDPSI <- L0Learn.fit(X, y, penalty="L0L2", maxSuppSize = 6, loss = "SquaredError", 
                    algorithm = "CDPSI", 
                    nGamma=5, gammaMin=0.1, gammaMax = 5,
                    scaleDownFactor = 0.2, intercept = TRUE)

#Summary of L0Learn.fit was printed out to show different lambda values at different suppsize
print(L0L2_CDPSI)

#The coefficients were extracted at appropriate lamda values (e.g. lambda = 0.0325142)
coef(L0L2_CDPSI, lambda = 0.000575243, gamma = 5.0000000) 
#The above represents the lambda value for suppsize 

#3. APPLYING THE L0-REGULARIZED REGRESSION MODEL
#The fitted model was applied on X to predict the response, predict_sss
predict_sss <- predict(L0L2_CDPSI, newx = X, lambda = 0.000575243, gamma = 5.0000000) 
print(predict_sss)

#The regularization path was plotted for a given gamma value, which varies with each experiment.
plot(L0L2_CDPSI, gamma = 5.0000000, showLines = TRUE)
 

# Appendix 4: Pearson’s Correlation Algorithm
#The relevant libraries in the R 4.4.1/ R-studio 2024.04.2+764 were imported to achieve the tasks in #this appendix.
#The lines of codes in this particular appendix are relevant to subsection 
#3.6 Determination of PPVs importance and collinearity with L0 models and algorithms
#A pairwise Pearson’s correlation analysis, which involved creation of a data frame containing pairs of predictors and their correlation values with collinear library and a function cor_df was performed to validate the assumption that a relatively perfect collinearity exits between any 2 PPVs if the #coefficient values of at least their #first 4 digits are exactly the same in the L0 models.

#1. DATA LOADING
#The appropriate library was imported.
library(collinear)

#The data file (.csv) with 7 variables sss, sst, ws, hws, precip, sla, & adt was uploaded interactively.
data <- read.csv(file.choose(), header = T)

#The data structure was viewed.
str(data)

#The 7 variables were order in the view as appropriate to create a new data frame (data_var) that replaces the previous data frame (data) above.
data_var <- data[, c("sss", "ws", "hws", "sst", "adt", "sla", "precip")] 

#The column and first few observations were viewed.
dplyr::glimpse(data_var)

#The response variable was stored as
data_var_sss <- data_var$sss

#The predictor variables were stored as
data_var_pred <- c("ws", "hws", "sst", "adt", "sla", "precip")

#2. COLLINEARITY ARGUMENT
#The argument for Collinearity was declared
collAss_predictors <- collinear(
  df = data_var,
  response = "sss",
  predictors = data_var_pred,
  preference_order = NULL,
  max_cor = 0.5,
  max_vif = 2.5,
)

#3. PAIRWISE PEARSON'S CORRELATION ANALYSIS
#Validated the assumption that a relatively perfect collinearity exits between any 2 PPVs if the #coefficient values of at #least their #first 4 digits are exactly the same in the L0 models.
collAssVal_predictors <- cor_df(
  df = data_var,
  response = "sss",
  predictors = data_var_pred,
  cor_method = "pearson",
  encoding_method = "mean"
)

head(collAssVal_predictors)
 

# Appendix 5: Machine Learning Metrics Algorithm
#The relevant library in the R 4.1.3/ R-studio #2022.02.3-492 was imported to achieve the tasks in this #appendix.
#The lines of codes in this particular appendix are relevant to subsections 
#3.9 Validation of the SSS forecasts

#1. DATA LOADING
#Uploaded the sss data in .csv file interactively using
data_pred_obs_sss <- read.csv(file.choose(), header = TRUE, stringsAsFactors = FALSE)

#Stored the uploaded data in a new data frame  as appropriate
data_sss <- data_pred_obs_sss[, c("sss_pred", "sss")]

#2. COMPUTING THE ERROR METRICS FOR VALIDATION OF THE SSS FORECASTS
#For the Validation of the SSS forecasts in 3.9, in relation to the observed SSS, the RMSE, MAE and #MAPE of the predicted SSS by LASSO for Jan.-Dec. 2021 were computed by #importing the MLmetrics library.
library(MLmetrics)

#The observed and predicated values were stored as appropriate using
sss_obs1 <- data_sss$sss
sss_pred1 <- data_sss$sss_pred

#The error metrics were computed using
RMSE(sss_pred1,sss_obs1)
MAE(sss_pred1,sss_obs1)
MAPE(sss_pred1,sss_obs1)*100

###The End###
