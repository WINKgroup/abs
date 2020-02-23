import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#### HELPER FUNCTIONS ####
def calcPriceQuantiles(series, period):
    if period.isna():
        return (np.nan, np.nan)
    tmpSeries = series.resample(rule=period).mean().interpolate()
    tmpSeries = tmpSeries.diff()
    loss, gain = tmpSeries.quantile([.05, .95])
    return (loss, gain)

def appendRiskValues(riskDf, df):
    isIn = df.iat[0,0]
    bidPeriod = (df.dropna(subset=["Bid Price"]))['Datetime'].diff().mean()
    askPeriod = (df.dropna(subset=["Ask Price"]))['Datetime'].diff().mean()
    tempDf = df.set_index('Datetime')
    riskDf.at[isIn, 'Bid Period'] = bidPeriod
    riskDf.at[isIn, 'Ask Period'] = askPeriod
    riskDf.at[isIn, 'Bid Loss'], riskDf.at[isIn, 'Bid Gain'] = calcPriceQuantiles(tempDf['Bid Price'], bidPeriod)
    riskDf.at[isIn, 'Ask Loss'], riskDf.at[isIn, 'Ask Gain'] = calcPriceQuantiles(tempDf['Ask Price'], askPeriod)

#### CLEANING ####
df = pd.read_csv('abs.csv', delimiter=";")
df = df.iloc[:, [0,4,5,6,7,11,12]]

print(df[['Isin', 'Ticker']].groupby('Isin').nunique().max())
del df['Ticker']

df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values(by="Datetime")
df['Bid Price'] = df['Bid Price'].str.replace(',','.').astype('float')
df['Ask Price'] = df['Ask Price'].str.replace(',','.').astype('float')
df['Bid Size'] = df['Bid Size'].str.replace(',','.').astype('float')
df['Ask Size'] = df['Ask Size'].str.replace(',','.').astype('float')


#### EXPLORATION ####
counts = df.groupby('Isin').count()
print(counts.sort_values(by="Bid Price"), counts.sort_values(by="Ask Price"))

df_XS0268642161 = df[df['Isin'] == 'XS0268642161']

plt.plot('Datetime', "Bid Price", data=df_XS0268642161, label='Bid')
plt.plot('Datetime', "Ask Price", data=df_XS0268642161, label='Ask')
plt.xlabel('Date')
plt.ylabel('Prezzo')
plt.title('XS0268642161')
plt.legend()
plt.show()


df_XS0260784318 = df[df['Isin'] == 'XS0260784318']

plt.plot('Datetime', "Bid Price", data=df_XS0260784318, label='Bid')
plt.plot('Datetime', "Ask Price", data=df_XS0260784318, label='Ask')
plt.xlabel('Date')
plt.ylabel('Prezzo')
plt.title('XS0260784318')
plt.legend()
plt.show()



#### ELABORATION ####

# Build "isIn" set of ids, list of DFs for each isIn
isinSet = set()
isinSet.update(list(df['Isin']))
dfList = [df[df['Isin'] == isin] for isin in isinSet]

# Elaborate riskDf
riskColumns = ['isIn', 'Bid Period', 'Ask Period', 'Bid Loss', 'Bid Gain', 'Ask Loss', 'Ask Gain']
riskDf = pd.DataFrame(columns=riskColumns).set_index('isIn')
for i in range(0, len(dfList)):
    appendRiskValues(riskDf, dfList[i])
    


