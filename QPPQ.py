
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import scipy
from scipy import stats 
from scipy.stats import norm
import seaborn as sns
import math

#DOCUMENTATION
# Units: MM3

### What to change: 
# Obs = gage that has more data and will serve as a template
# Pre = gage that we want to predict
# file_for_template is the gauge you'll use as the one with the most data, the one you'll use as your template
# file_to_predict is the gauge that will be estimated using the file_for_template


##=========================================================================================
## Read csv files
##=========================================================================================

folder = '/Users/lauraelisa/Desktop/RioGrande/QPPQ/input/'
results = '/Users/lauraelisa/Desktop/RioGrande/QPPQ/results/'
file_for_template = 'Daily_Streamflow_DelRio_finalcompleted'
file_to_predict = 'Daily_Streamflow_Anzalduas_original'


##Set Index and Open dataframes

df_obs = pd.read_csv('{}/{}.csv'.format(folder,file_for_template),index_col=0)
df_obs.columns = ['streamflow_for_template']
df_obs.set_index(pd.date_range(start='1/1/1900', end='31/12/2018', freq='D'), inplace=True, drop=True) #end='12/31/2005', end='1/1/2006',



df_pre = pd.read_csv('{}/{}.csv'.format(folder,file_to_predict),index_col=0)
df_pre.columns = ['streamflow_to_predict']
df_pre.set_index(pd.date_range(start='1/1/1900', end='31/12/2018', freq='D'), inplace=True, drop=True) #end='12/31/2005', end='1/1/2006',


## =========================================================================================
## Flow Duration Curves (FDC) 
## =========================================================================================

obs_sort = df_obs.sort_values(by=['streamflow_for_template'], axis=0, ascending=True)
obs_sort.dropna(axis=0, inplace=True)
exceedence = 1.-np.arange(1.,len(obs_sort) + 1.)/len(obs_sort)


pred_sort = df_pre.sort_values(by=['streamflow_to_predict'], axis=0, ascending=True)
pred_sort.dropna(axis=0, inplace=True)
exceedence_2 = 1.-np.arange(1.,len(pred_sort) + 1.)/len(pred_sort)

# #Plots
# plt.plot(exceedence*100,obs_sort)
# plt.xlabel("Exceedence [%]")
# plt.ylabel("Flow rate")
# plt.show()

# plt.plot(exceedence_2*100,pred_sort)
# plt.xlabel("Exceedence [%]")
# plt.ylabel("Flow rate")
# plt.show()


## =========================================================================================
## ###Stats,
## =========================================================================================

weibull_obs = pd.DataFrame(scipy.stats.mstats.plotting_positions(obs_sort, alpha=0, beta=0))
cunnae_obs = pd.DataFrame(scipy.stats.mstats.plotting_positions(obs_sort, alpha=0.4, beta=0.4))
blom_obs = pd.DataFrame(scipy.stats.mstats.plotting_positions(obs_sort, alpha=3/8, beta=3/8))

weibull_pred = pd.DataFrame(scipy.stats.mstats.plotting_positions(pred_sort, alpha=0, beta=0))
cunnae_pred = pd.DataFrame(scipy.stats.mstats.plotting_positions(pred_sort, alpha=0.4, beta=0.4))
blom_pred = pd.DataFrame(scipy.stats.mstats.plotting_positions(pred_sort, alpha=3/8, beta=3/8))


print('Weibull Obs results',weibull_obs.tail(1)) #Print last element of the array
print('Cunnane Obs results', cunnae_obs.tail(1))
print('Blom Obs results', blom_obs.tail(1)) 

print('Weibull Predicted results',weibull_pred.tail(1)) #Print last element of the array
print('Cunnane Predicted results', cunnae_pred.tail(1))
print('Blom Predicted results', blom_pred.tail(1)) 

### Values for alpha and beta depending on the type of distribution
# Typical values for alpha and beta are:
# (0,1) : p(k) = k/n, linear interpolation of cdf (R, type 4)
# (.5,.5) : p(k) = (k-1/2.)/n, piecewise linear function (R, type 5)
# (0,0) : p(k) = k/(n+1), Weibull (R type 6)
# (1,1) : p(k) = (k-1)/(n-1), in this case, p(k) = mode[F(x[k])]. That’s R default (R type 7)
# (1/3,1/3): p(k) = (k-1/3)/(n+1/3), then p(k) ~ median[F(x[k])]. The resulting quantile estimates are approximately median-unbiased regardless of the distribution of x. (R type 8)
# (3/8,3/8): p(k) = (k-3/8)/(n+1/4), Blom. The resulting quantile estimates are approximately unbiased if x is normally distributed (R type 9)
# (.4,.4) : approximately quantile unbiased (Cunnane)
# (.35,.35): APL, used with PWM
# (.3175, .3175): used in scipy.stats.probplot
      

##=========================================================================================
## Make DF with complete data
##=========================================================================================

## Observed streamflow
anchor = pd.DataFrame(obs_sort)
anchor.reset_index(inplace = True)

obs_gage = pd.concat([anchor, cunnae_obs], axis=1, join="inner")
obs_gage.columns = ['Date','Streamflow_obs', 'P(x)_Cunnae']
obs_gage = obs_gage.sort_values(by=['Date'], axis=0, ascending=True)
obs_gage = obs_gage.reset_index()


## Predicted streamflow
rope = pd.DataFrame(pred_sort)
rope.reset_index(inplace = True)

pred_gage = pd.concat([rope, cunnae_pred], axis=1, join="inner")
pred_gage.columns = ['Date','Streamflow_predicted', 'P(x)_Cunnae']
pred_gage = pred_gage.sort_values(by=['Date'], axis=0, ascending=True)
pred_gage = pred_gage.reset_index()


obs_gage = obs_gage.sort_values(by=['P(x)_Cunnae'], axis=0, ascending=True)
pred_gage = pred_gage.sort_values(by=['P(x)_Cunnae'], axis=0, ascending=True)


##=========================================================================================
## Correlation
##=========================================================================================
## CORRELATION using VLOOKUP!!! is done with merge_asof to correlate the obs gage to the predicted gage
qppq = pd.merge_asof(obs_gage,pred_gage[['P(x)_Cunnae','Streamflow_predicted']], on='P(x)_Cunnae', direction='nearest')


##=========================================================================================
## Results QPPQ For predicted
##=========================================================================================
## in the gage to be predicted, if there is any gap, then assign the values of QPPQ.
qppq_predicted = pd.DataFrame(qppq[['Date','Streamflow_predicted']]).sort_values(by=['Date'], axis=0, ascending=True)

df_pre.reset_index(inplace = True)
df_pre.columns = ['Date','Streamflow_obs']


streamflow_predicted = pd.merge(df_pre, qppq_predicted, on='Date')
streamflow_predicted["Streamflow_complete"] = streamflow_predicted["Streamflow_obs"].fillna(streamflow_predicted["Streamflow_predicted"])
result = pd.DataFrame(streamflow_predicted[['Date','Streamflow_complete']])


# 


index = pd.date_range(start='1/1/1900', end='31/12/2018')
#columns =id_nombre
#df = pd.DataFrame(np.nan,index=index, columns = columns)
df2 = pd.DataFrame(index=index)

result2 = df2.join(result, how='left') #add index from 1900 to 2010
result2.to_csv('{}{}_qppq.csv'.format(results,file_to_predict)) #Save dictionaries to CSV files



##=========================================================================================
## Save
##=========================================================================================

result.to_csv('{}{}_qppq_recalibrated.csv'.format(results,file_to_predict)) #Save dictionaries to CSV files



# ##=========================================================================================
# ## check types
# ##=========================================================================================


# print(df_pre.dtypes)
# print(qppq_predicted.dtypes)
# print(streamflow_predicted.dtypes)
