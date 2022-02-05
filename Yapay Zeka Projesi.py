#!/usr/bin/env python
# coding: utf-8

# In[146]:


import pandas as pd
import matplotlib.pyplot as plt 
from sklearn import linear_model
import seaborn as sns

df=pd.read_csv(r"C:\Users\guzel\Desktop\yapay zeka\multilinearregression.csv",sep = ";")


# In[147]:


df


# In[148]:


df.info() #Veri özellikleri


# In[149]:


df.columns


# In[150]:


df.describe() #istatistiksel değerler


# In[ ]:


#Eksik veri ayıklaması


# In[151]:


for column in df.columns[1:-1]:
    df[column].fillna(value=df[column].mean(), inplace=True) 


# In[152]:


df.columns[1:-1]


# In[153]:


df.isna().sum()


# In[154]:


df.head()


# In[155]:


df.groupby("konum").agg(["min","max","std","mean"]) 


# In[ ]:


#VERİ GÖRSELLEŞTİRME


# In[156]:


df.konum.value_counts().plot.barh() 


# In[157]:


sns.lmplot(x='fiyat', y='alan', data=df)


# In[158]:


sns.lmplot(x='fiyat', y='odasayisi', data=df)


# In[159]:


sns.lmplot(x='fiyat', y='binayasi', data=df)


# In[160]:


sns.lmplot(x = 'alan', y = 'fiyat', fit_reg = False, hue = 'konum', data = df)


# In[161]:


reg = linear_model.LinearRegression()
reg.fit(df[['alan', 'odasayisi', 'binayasi']], df['fiyat'])
hesapla = reg.predict([[230,4,10], [230,6,0], [355,3,20]])
print(hesapla) 


# In[162]:


reg.coef_ #regresyon katsayısı


# In[163]:


reg.intercept_ #kesim noktası


# In[164]:


import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from sklearn import model_selection


# In[165]:


df.columns


# In[166]:


df = df[["alan","odasayisi","binayasi","fiyat"]]


# In[167]:


df


# In[168]:


#bağımlı ve bağımsız değişkenlerin ayrılması
X = df.drop(["fiyat"], axis = 1)
y = df["fiyat"] 


# In[180]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 144) #Eğitim ve test verisi ayrılması


# In[181]:


params = {"colsample_bytree":[0.4,0.5,0.6],
         "learning_rate":[0.01,0.02,0.09],
         "max_depth":[2,3,4,5,6],
         "n_estimators":[100,200,500,2000]}


# In[182]:


xgb = XGBRegressor()


# In[171]:


grid = GridSearchCV(xgb, params, cv = 10, n_jobs = -1, verbose = 2)


# In[172]:


xgb1 = XGBRegressor(colsample_bytree = 0.5, learning_rate = 0.09, max_depth = 4, n_estimators = 2000) #En uygun parametreler


# In[173]:


model_xgb = xgb1.fit(X_train, y_train) #Modelin eğitilmesi


# In[174]:


model_xgb.predict(X_test)[15:20] #tahmin yapılması


# In[175]:


y_test[15:20] #Tahmin ve gerçek değerlerin karşılaştırılması


# In[183]:


model_xgb.score(X_test, y_test) #Modelin skoru 0-1


# In[184]:


model_xgb.score(X_train, y_train)


# In[185]:


np.sqrt(-1*(cross_val_score(model_xgb, X_test, y_test, cv=10, scoring='neg_mean_squared_error'))).mean() #Hata oranı


# In[ ]:




