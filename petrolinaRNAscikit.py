# -*- coding: utf-8 -*-
"""
Created on Mon May 11 00:04:46 2020

@author: leand
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

xlsx60=pd.ExcelFile("Pet2014train.xlsx")
df=pd.read_excel(xlsx60, "Planilha1")
m=df.shape[0]
df1=df.radext
df2=df.ghiext
df3=df.temp
df7=df.umidade

xlsx40=pd.ExcelFile("Pet2014test.xlsx")
df4=pd.read_excel(xlsx40, "Planilha1")
df5=df4.ghiext
df6=df4.temp
df8=df4.umidade
df13=df4.radext

xlsx70=pd.ExcelFile("Pet2010predição.xlsx")
df9=pd.read_excel(xlsx70, "Planilha1")
df10=df9.ghiext
df11=df9.temp
df12=df9.umidade
df14=df9.radext

X_train = np.array(([df2,df3,df7]), dtype=float)
X_train=X_train.transpose()



X_test = np.array(([df5,df6,df8]), dtype=float)
X_test = X_test.transpose()

X_predict = np.array(([df10,df11,df12]), dtype=float)
X_predict = X_predict.transpose()

y_train = np.array(([df1]), dtype=float)   
y_train = y_train.transpose().ravel()

y_test = np.array(([df13]), dtype=float)
y_test = y_test.transpose()

y_predict = np.array(([df14]), dtype=float)
y_predict = y_predict.transpose()








reg = MLPRegressor(hidden_layer_sizes=(3, 3, 3),  activation='relu', solver='adam',alpha=0.001,batch_size='auto',
               learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
              random_state=0, tol=0.00001, verbose=False, warm_start=False, momentum=0.9,
               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
               epsilon=1e-08)

reg = reg.fit(X_train, y_train)

test_y = reg.predict(X_test)
predict_y = reg.predict(X_predict)

rmse = np.sqrt(mean_squared_error(y_test,test_y))


fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(y_test, test_y, s=10, c='b', marker="s", label='real')
plt.axis([0.2,0.8,0.2,0.8])


plt.show()

print(rmse)

from matplotlib import pyplot as plt
plt.close('all') #fecha todos os gráficos para nao ocorrer interferencia
predição=predict_y   #pega todas as linhas e segunda coluna do arquivo
glo_avg=df14
GHI=df10

fig = plt.figure(0,figsize=(7.5,4))


plt.plot(glo_avg, glo_avg, '+g')
plt.plot(glo_avg,predição, '+r')
plt.plot(glo_avg,GHI, '+k')

coef=np.polyfit(glo_avg, GHI,1)
regr=np.poly1d(coef)
reta=regr(glo_avg)
plt.plot(glo_avg,reta,'k')

coef1=np.polyfit(glo_avg,predição, 1)
regr1=np.poly1d(coef1)
reta1=regr1(glo_avg)
plt.plot(glo_avg,reta1,'r')

coef2=np.polyfit(glo_avg,glo_avg,1)
regr2=np.poly1d(coef2)
reta2=regr2=(glo_avg)
plt.plot(glo_avg,reta2,'g')

plt.xlabel('Valores observados normalizados', fontsize='13')
plt.ylabel('Valores estimados normalizados',fontsize='13')
plt.axis([0.1,0.9,0.1,0.9])

fig.savefig('Fig1.png', bbox_inches = 'tight', dpi = 300) #salva em qualquer formato de imagem

fig2=plt.figure(figsize=(15, 12))
plt.subplot(2, 3, 3)
plt.ylabel('RNA Validation')
plt.xlabel('Observations')
plt.title("Scatter Plot")

#plt.tight_layout()

plt.scatter(y_predict, predict_y, c="g")
zz=np.linspace(0,1,2) 
plt.plot(zz,zz,'k--') # identity line
plt.xlim(0,1)
plt.ylim(0,1)


plt.subplot(2, 3, 2)
plt.ylabel('RNA Test')
plt.xlabel('Observations')
plt.title("Scatter Plot")
plt.scatter(y_test, test_y, c="b")
plt.plot(zz,zz,'k--') # identity line
plt.xlim(0,1)
plt.ylim(0,1)




fig2.savefig("scatter.png", dpi=300) # salva gráficos