import pandas as pd
import pyodbc

#import regular season stats from MLB teams who got into postseason during 2012-2019
#items include Tm, BatAge, PA, AB, R, H, 2B, 3B, HR, RBI, SB, CS, BB, SO, BA, OBP, SLG, OPS, TB, GDP
#total rows are 8(years)*10(teams each year)=80

sql_conn = pyodbc.connect('''DRIVER={ODBC Driver 13 for SQL Server};
                            SERVER=ALLENHO\MSSQLSERVER002;
                            DATABASE=Playoffbound;
                            Trusted_Connection=yes''') 
query = '''
select Tm, BatAge, PA, AB, R, H, 2B, 3B, HR, RBI, SB, CS, BB, SO, BA, OBP, SLG, OPS, TB, GDP
from [dbo].['19B$']
where Tm in ('WSN','LAD','MIL','ATL','STL','HOU','NYY','MIN','TBR','OAK')
UNION ALL
select Tm, BatAge, PA, AB, R, H, 2B, 3B, HR, RBI, SB, CS, BB, SO, BA, OBP, SLG, OPS, TB, GDP
from [dbo].['18B$']
where Tm in ('BOS','LAD','MIL','ATL','CHC','HOU','NYY','CLE','COL','OAK')
UNION ALL
select Tm, BatAge, PA, AB, R, H, 2B, 3B, HR, RBI, SB, CS, BB, SO, BA, OBP, SLG, OPS, TB, GDP
from [dbo].['17B$']
where Tm in ('BOS','LAD','COL','WSN','CHC','HOU','NYY','CLE','ARI','MIN')
UNION ALL
select Tm, BatAge, PA, AB, R, H, 2B, 3B, HR, RBI, SB, CS, BB, SO, BA, OBP, SLG, OPS, TB, GDP
from [dbo].['16B$']
where Tm in ('TOR','CLE','BOS','BAL','TEX','NYM','CHC','LAD','WSN','SFG')
UNION ALL
select Tm, BatAge, PA, AB, R, H, 2B, 3B, HR, RBI, SB, CS, BB, SO, BA, OBP, SLG, OPS, TB, GDP
from [dbo].['15B$']
where Tm in ('TOR','KCR','HOU','NYY','TEX','NYM','CHC','LAD','STL','PIT')
UNION ALL
select Tm, BatAge, PA, AB, R, H, 2B, 3B, HR, RBI, SB, CS, BB, SO, BA, OBP, SLG, OPS, TB, GDP
from [dbo].['14B$']
where Tm in ('BAL','KCR','OAK','LAA','DET','WSN','STL','LAD','PIT','SFG')
UNION ALL
select Tm, BatAge, PA, AB, R, H, 2B, 3B, HR, RBI, SB, CS, BB, SO, BA, OBP, SLG, OPS, TB, GDP
from [dbo].['13B$']
where Tm in ('BOS','TBR','OAK','CLE','DET','ATL','STL','LAD','PIT','CIN')
UNION ALL
select Tm, BatAge, PA, AB, R, H, 2B, 3B, HR, RBI, SB, CS, BB, SO, BA, OBP, SLG, OPS, TB, GDP
from [dbo].['12B$']
where Tm in ('TEX','BAL','OAK','NYY','DET','ATL','STL','SFG','WSN','CIN')
'''
df = pd.read_sql(query, sql_conn)

#stored as df_post
df_post = df

#import regular season stats from MLB teams who DIDN'T get into postseason during 2012-2019
#items are the same as above
#total rows are 8(years)*20(teams each year)=160
sql_conn = pyodbc.connect('''DRIVER={ODBC Driver 13 for SQL Server};
                            SERVER=ALLENHO\MSSQLSERVER002;
                            DATABASE=Playoffbound;
                            Trusted_Connection=yes''') 
query = '''
select Tm, BatAge, PA, AB, R, H, 2B, 3B, HR, RBI, SB, CS, BB, SO, BA, OBP, SLG, OPS, TB, GDP
from [dbo].['19B$']
where Tm is not null and Tm not in ('WSN','LAD','MIL','ATL','STL','HOU','NYY','MIN','TBR','OAK', 'LgAvg')
UNION ALL
select Tm, BatAge, PA, AB, R, H, 2B, 3B, HR, RBI, SB, CS, BB, SO, BA, OBP, SLG, OPS, TB, GDP
from [dbo].['18B$']
where Tm is not null and Tm not in ('BOS','LAD','MIL','ATL','CHC','HOU','NYY','CLE','COL','OAK', 'LgAvg')
UNION ALL
select Tm, BatAge, PA, AB, R, H, 2B, 3B, HR, RBI, SB, CS, BB, SO, BA, OBP, SLG, OPS, TB, GDP
from [dbo].['17B$']
where Tm is not null and Tm not in ('BOS','LAD','COL','WSN','CHC','HOU','NYY','CLE','ARI','MIN', 'LgAvg')
UNION ALL
select Tm, BatAge, PA, AB, R, H, 2B, 3B, HR, RBI, SB, CS, BB, SO, BA, OBP, SLG, OPS, TB, GDP
from [dbo].['16B$']
where Tm is not null and Tm not in ('TOR','CLE','BOS','BAL','TEX','NYM','CHC','LAD','WSN','SFG', 'LgAvg')
UNION ALL
select Tm, BatAge, PA, AB, R, H, 2B, 3B, HR, RBI, SB, CS, BB, SO, BA, OBP, SLG, OPS, TB, GDP
from [dbo].['15B$']
where Tm is not null and Tm not in ('TOR','KCR','HOU','NYY','TEX','NYM','CHC','LAD','STL','PIT', 'LgAvg')
UNION ALL
select Tm, BatAge, PA, AB, R, H, 2B, 3B, HR, RBI, SB, CS, BB, SO, BA, OBP, SLG, OPS, TB, GDP
from [dbo].['14B$']
where Tm is not null and Tm not in ('BAL','KCR','OAK','LAA','DET','WSN','STL','LAD','PIT','SFG', 'LgAvg')
UNION ALL
select Tm, BatAge, PA, AB, R, H, 2B, 3B, HR, RBI, SB, CS, BB, SO, BA, OBP, SLG, OPS, TB, GDP
from [dbo].['13B$']
where Tm is not null and Tm not in ('BOS','TBR','OAK','CLE','DET','ATL','STL','LAD','PIT','CIN', 'LgAvg')
UNION ALL
select Tm, BatAge, PA, AB, R, H, 2B, 3B, HR, RBI, SB, CS, BB, SO, BA, OBP, SLG, OPS, TB, GDP
from [dbo].['12B$']
where Tm is not null and Tm not in ('TEX','BAL','OAK','NYY','DET','ATL','STL','SFG','WSN','CIN', 'LgAvg')
'''
df = pd.read_sql(query, sql_conn)

#stored as df_npost
df_npost = df

#add each dataframe a new column named POST, which imply whether the team made the postseason
df_post['POST']= 1
df_npost['POST']= 0

#append two dataframes together
df_com=df_post.append(df_npost)

#-----using logistic model to see team season OBP vs making the postseason that year or not-------------------------------

# Load libraries and functions
import statsmodels.api as sm
from statsmodels.formula.api import glm
import numpy as np

# Define the formula of the logistic model to see how well team regular season OBP predict postseason birth
model_formula = 'POST ~ OBP'
# Define the correct probability distribution and the link function of the response variable
link_function = sm.families.links.logit
model_family = sm.families.Binomial(link = link_function)

# Fit the model
OBP_fit = glm(formula = model_formula, 
                data = df_com, 
                family = model_family).fit()
# View the results of the OBP_fit model
print(OBP_fit.summary())

# Extract coefficients from the fitted model OBP_fit
intercept, slope = OBP_fit.params

# Print coefficients
print('Intercept =', intercept)
print('Slope =', slope)

# Extract and print confidence intervals
print(OBP_fit.conf_int())

# Compute the multiplicative effect on the odds
print('Odds: \n', np.exp(OBP_fit.params))

# Define x at 0.333
x = 0.333

# Compute and print the estimated probability
est_prob = np.exp(intercept + slope*x)/(1+np.exp(intercept + slope*x))
print('Estimated probability at x = 0.333: ', round(est_prob, 4))

# Compute the slope of the tangent line for parameter beta at x
slope_tan = slope * est_prob * (1 - est_prob)
print('The rate of change in probability: ', round(slope_tan,4))

# Estimated covariance matrix: OBP_cov
OBP_cov = OBP_fit.cov_params()
print(OBP_cov)

# Compute standard error (SE): std_error
std_error = np.sqrt(OBP_cov.loc['OBP', 'OBP'])
print('SE: ', round(std_error, 4))

# Compute Wald statistic
wald_stat = slope/std_error
print('Wald statistic: ', round(wald_stat,4))

# Extract and print confidence intervals
print(OBP_fit.conf_int())

# Compute confidence intervals for the odds
print(np.exp(OBP_fit.conf_int()))

import seaborn as sns
import matplotlib.pyplot as plt

# Plot OBP and POST and add overlay with the logistic fit
sns.regplot(x = 'OBP', y = 'POST', 
            y_jitter=0.03,
            data = df_com, 
            logistic = True,
            ci = None)
# Display the plot
plt.show()


#-----assume there is now a new 2020 dataset named OBP_test, we can test how well it fits out model

# Compute predictions for the test sample OBP_test and save as prediction
prediction = OBP_fit.predict(exog = OBP_test)

# Add prediction to the existing data frame OBP_test and assign column name prediction
OBP_test['prediction'] = prediction

# Examine the first 5 computed predictions
print(OBP_test[['POST', 'OBP', 'prediction']].head())

# Define the cutoff
cutoff = 0.5

# Compute class predictions: y_prediction
y_prediction = np.where(prediction > cutoff, 1, 0)

# Assign actual class labels from the test sample to y_actual
y_actual = OBP_test['POST']

# Compute the confusion matrix using crosstab function
conf_mat = pd.crosstab(y_actual, y_prediction,
					   rownames=['Actual'], 
                  	   colnames=['Predicted'], 
                       margins = True)

# Print the confusion matrix
print(conf_mat)

#-----above code are examples of team season OBP vs making the postseason that year or not
#-----the code can all be used to see the logistic model between team season stats vs teams making the postseason or not 

#-----below are multivariate logistic regression---------------------------------------------
#-----using BA, OBP, SLG, OPS as example 
# Define model formula
formula = 'POST ~ BA+OBP+SLG+OPS'

# Fit GLM
model = glm(formula, data = df_com, family = sm.families.Binomial()).fit()

# Print model summary
print(model.summary())

# Import functions
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Get variables for which to compute VIF and add intercept term
X = df_com[['BA', 'OBP', 'SLG', 'OPS']]
X['Intercept'] = 1

# Compute and view VIF
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# View results using print
print(vif)

# Compare deviance of null and residual model
diff_deviance = model.null_deviance - model.deviance

# Print the computed difference in deviance
print(diff_deviance)

# define formula_BA
formula_BA = 'POST ~ BA'

# Fit GLM
model_BA = glm(formula_BA, data = df_com, family = sm.families.Binomial()).fit()

# Compare deviance of null and residual model
diff_deviance = model_BA.null_deviance - model_BA.deviance

# Print the computed difference in deviance
print('Adding BA to the null model reduces deviance by: ', 
      round(diff_deviance,3))

# define formula_OBP
formula_OBP = 'POST ~ OBP'

# Fit GLM
model_OBP = glm(formula_OBP, data = df_com, family = sm.families.Binomial()).fit()

# Compute the difference in adding OBP variable
diff_deviance_1 = model_OBP.deviance - model_BA.deviance

# Print the computed difference in deviance
print('Adding OBP to the BA model reduces deviance by: ', 
      round(diff_deviance_1,3))

# define formula_SLG
formula_SLG = 'POST ~ SLG'

# Fit GLM
model_SLG = glm(formula_SLG, data = df_com, family = sm.families.Binomial()).fit()

# Compute the difference in adding BA variable
diff_deviance_2 = model_SLG.deviance - model_BA.deviance

# Print the computed difference in deviance
print('Adding SLG to the BA model reduces deviance by: ', 
      round(diff_deviance_2,3))

# define formula_OPS
formula_OPS = 'POST ~ OPS'

# Fit GLM
model_OPS = glm(formula_OPS, data = df_com, family = sm.families.Binomial()).fit()

# Compute the difference in adding BA variable
diff_deviance_3 = model_OPS.deviance - model_BA.deviance

# Print the computed difference in deviance
print('Adding OPS to the BA model reduces deviance by: ', 
      round(diff_deviance_3,3))

# Import function dmatrix()
from patsy import dmatrix
import numpy as np

# Construct model matrix with OBP
model_matrix = dmatrix('OBP', data = df_com, return_type = 'dataframe')
print(model_matrix.head())

# Construct model matrix with OBP and OPS
model_matrix_1 = dmatrix('OBP+OPS', data = df_com, return_type = 'dataframe')
print(model_matrix_1.head())


# Construct model matrix for OBP with log transformation
dmatrix('np.log(OBP)', data = df_com,
       return_type = 'dataframe').head()


# Define model formula
formula = 'POST ~ np.log(OBP)'
# Fit GLM
model_log_OBP = glm(formula, data = df_com, 
                     family = sm.families.Binomial()).fit()
# Print model summary
print(model_log_OBP.summary())
