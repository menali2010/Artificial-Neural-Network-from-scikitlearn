# Artificial-Neural-Network-from-scikitlearn

Method for generating time series of daily solar irradiance from one year of station data (temperature, humidity) combined with values 
produced by a satellite model.

#Artificial Neural Networks (ANNs) are systems composed of neurons that solve mathematical functions (both linear and nonlinear). ANNs are a
tool capable of storing knowledge from training (examples) and can be used in various areas, such as prediction models, pattern recognition,
and other human applications. 

#To develop an ANN, it is necessary to execute code functions that train the network. In this stage, examples are provided to the network, 
which initially adjusts its synaptic weights and biases randomly and gradually refines these values through various functions until it 
extracts the best combinations to represent the data. Subsequently, these values are fixed and used to generate solutions for new input data.

#The objective of this code is to utilize an existing method for generating time series of daily solar irradiance from one year of field data 
acquisition combined with values produced by the BRASIL-SR model and used in the Brazilian Solar Energy Atlas. 

#To start creating the code, it is necessary to process the data obtained from the stations and the satellite, normalizing them to the range
between 0 and 1. It is also necessary to group them into three files to import this information into the algorithm. The training file 
(Pet2014train) should contain 66.6% of the data corresponding to one year, with approximately 20 data points for each month. The second 
file is the testing file (Pet2014test), which contains the remaining 33.3% of data from the year used for training. The third file is the 
prediction file (Pet2010prediction), which contains another full year of data. 

#Initially, the data is grouped into three distinct spreadsheets used for training, testing, and prediction. Due to the storage format of 
the values (Excel), the pandas tool was used for importing. After the import, each set of values was saved in different data frames for later
application in matrices. The values inserted into X are the input data of the system, which include satellite-estimated irradiation, surface
temperature, and relative humidity. The data inserted into y are the output values, the expected result, which is the measured irradiation 
at the station. The last variable, xPredicted, represents the data that will be used for prediction, having the same properties as the 
variable X. However, the code should predict the output value these values will provide based on the training and testing steps using the 
variables X_train/X_test and y_train/y_test.


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

#_____________________________________________________________________________________________________

#After the creation of variables, we begin to use the code for the Artificial Neural Network. Within the Anaconda software, it is 
possible to use a ready-made library for artificial neural networks of various types, where you can choose the one that best fits the 
desired objectives, requiring only parameter changes. The library used is called Scikit-Learn, which is a package of various functions 
used in machine learning, in the Python language, and from it, we can import the ANN data package, specifically the Regression package.

#The chosen model is the Multi-Layer Perceptron (MLP) as it has the advantage of being able to learn non-linear models. Additionally, the 
MLPRegressor was used, which is an MLP that utilizes backpropagation to train the network in the best way and obtain the best output 
values. 

#Looking at the code below, you can see that the parameters have already been defined, based on the organized data. Some parameters were 
adjusted while others were left in their default values in the RNA. The following are the parameters used:

#->hidden_layer_sizes: Determines the number of neurons and the number of layers in the MLP. In this case, a 3-layer MLP with 1 neuron per
input variable was chosen.

#->activation: Refers to the activation function used. The 'relu' function was used as it yielded better results for daily values in the 
Scikit-Learn library. However, for hourly values, the hyperbolic tangent activation function showed better results.

#->solver: Determines the optimization algorithm for obtaining the network weights. For daily predictions, the 'adam' solver was used, 
which utilizes stochastic gradient descent.

#->learning_rate: Determines how the code behaves in updating the weights to achieve good results. In this case, the learning rate was 
set to 'constant', which means a constant learning rate dependent on the following factor.

#->learning_rate_init: Specifies the initial learning rate used to update the weights. The system is very sensitive to any changes in 
these values. For this data, a value of 0.01 was used.

#The remaining parameters were left at their default values in the library, as no significant or positive changes were observed in the 
results.

reg = MLPRegressor(hidden_layer_sizes=(3, 3, 3),  activation='relu', solver='adam',alpha=0.001,batch_size='auto',
               learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=1000, shuffle=True,
              random_state=0, tol=0.00001, verbose=False, warm_start=False, momentum=0.9,
               nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
               epsilon=1e-08)
               
#_____________________________________________________________________________________________________


#The next step refers to training the data, where we can use the "reg.fit" function. Afterwards, tests are performed to track the 
obtained error values (Root Mean Square Error - RMSE) throughout the use of the ANN. Simultaneously, predictions are made for a 
selected set of data, concluding the network.
 
#The part of the code related to graphs was created to visually compare the predicted results with the expected values and to establish
a relationship between them. It allows for better understanding and analysis of the performance of the neural network.
 
 
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

#_____________________________________________________________________________________________________

#Additionally, new graphs were created to generate trend lines, comparing the performance of different models with the measured values 
from the station. Scatter plots were also developed to visually analyze how the network performs in terms of data dispersion. These 
visualizations help to assess the accuracy and effectiveness of the neural network in capturing the underlying patterns in the data.

from matplotlib import pyplot as plt

plt.close('all') #Close all the plots to prevent interference.

predição=predict_y 

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


fig2.savefig("scatter.png", dpi=300)


