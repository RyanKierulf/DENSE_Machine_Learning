#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy
import pandas

#Her_Model_2014, command: ./gillespie -p ./paramsets.csv -c 10 -w 1 -t 300 -u 0.5

concentrations = pandas.read_csv("concentrations.csv") #concentrations is 599 x 141
rates = pandas.read_csv("rates.csv") #rates is 599 x 341
performance = pandas.read_csv("performance.csv") #performance is 599 x 1

concentrations_size = concentrations.shape
rates_size = rates.shape
performance_size = performance.shape

if (concentrations_size[0] !=  rates_size[0]) or (rates_size[0] != concentrations_size[0]) or (performance_size[0] != concentrations_size[0]):
    print("ERROR: number of rows not equal in rates, concentrations, and performance!")

X = pandas.concat([concentrations, rates], axis=1)
Y = performance

if (X.shape[0] != concentrations_size[0]) or (X.shape[1] != (concentrations_size[1] + rates_size[1])):
    print("ERROR: merge failed")     

#Parse X data, delete columns of only zeros
X = X.to_numpy()
Y = Y.to_numpy()
empty_column_indices = []
for i in range(0, X.shape[1]):
    is_empty = True
    for j in range(X.shape[0]):
        if (X[j][i] != 0):
            is_empty = False
    if (is_empty is True):
        empty_column_indices.append(i)
        
num_columns_not_empty = X.shape[1] - len(empty_column_indices) #72 columns not empty
new_X = numpy.zeros([X.shape[0], num_columns_not_empty])
non_empty_column_indices = []

for i in range(X.shape[1]):
    if i not in empty_column_indices:
        non_empty_column_indices.append(i)
    
for i in range(num_columns_not_empty):
    new_X[:,i] = X[:,non_empty_column_indices[i]]

X = new_X

#At this point, notice that some columns have only nan values- I don't see any with some nans and
#some numbers. Delete the columns of only nan values

nan_columns = []
for i in range(X.shape[1]):
    only_nans = True
    for j in range(X.shape[0]):
        if (numpy.isnan(X[j][i]) == False):
            only_nans = False
    if only_nans is True:
        nan_columns.append(i)
            
num_columns_not_nans = X.shape[1] - len(nan_columns)
new_X = numpy.zeros([X.shape[0], num_columns_not_nans])
non_nans_indices = []
for i in range(X.shape[1]):
    if i not in nan_columns:
        non_nans_indices.append(i)
        
for i in range(num_columns_not_nans):
    new_X[:,i] = X[:,non_nans_indices[i]]
        
X = new_X

#Now normalize data using min-max scaling: x(scaled) = (x - xmin)/(xmax - xmin)
#First find min and max of each column

mins = []
maxs = []

for i in range(X.shape[1]):
    maxs.append(numpy.amax(X[:,i]))
    mins.append(numpy.amin(X[:,i]))
    
#Notice that some columns are constant for each row. These columns are useless,
#and will produce problems when scaling, so delete them

ranges = []
for i in range(len(mins)):
    ranges.append(maxs[i] - mins[i])
    
constant_indices = []
for i in range(X.shape[1]):
    if (ranges[i] < 0.01):
        constant_indices.append(i)
        
num_columns_not_constant = X.shape[1] - len(constant_indices)
new_X = numpy.zeros([X.shape[0], num_columns_not_constant])
not_constant_indices = []

for i in range(X.shape[1]):
    if i not in constant_indices:
        not_constant_indices.append(i)
        
for i in range(num_columns_not_constant):
    new_X[:,i] = X[:,not_constant_indices[i]]
    
X = new_X

#Now that constant columns are gone, complete scaling. Need to find mins and maxes again:
mins = []
maxs = []

for i in range(X.shape[1]):
    maxs.append(numpy.amax(X[:,i]))
    mins.append(numpy.amin(X[:,i]))
    
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        X[i][j] = (X[i][j] - mins[j]) / (maxs[j] - mins[j])
        if (X[i][j] < 0) or (X[i][j] > 1):
            print("ERROR: Normalization Failed")
            
Y = Y.tolist()


# In[14]:


from sklearn import linear_model
import statistics

#Linear Regression
#https://towardsdatascience.com/simple-and-multiple-linear-regression-in-python-c928425168f9

lm = linear_model.LinearRegression()
model = lm.fit(X,Y)

predictions = lm.predict(X)
#print(predictions)
predictions = predictions.tolist()

coefficients = lm.coef_
print(coefficients)

if (len(Y) != len(predictions)):
    print("Error: Y and predictions of unequal size!")
    
abs_errors = []
for i in range(len(predictions)):
    prediction = predictions[i]
    performance_value = Y[i]
    #Prediction and performance value are both lists with one item:
    prediction = prediction[0]
    performance_value = performance_value[0]
    error = abs(prediction - performance_value)
    abs_errors.append(error)

mean_error = sum(abs_errors)/len(abs_errors)
#Make Y just a list, so pstdev can be called
for i in range(len(Y)):
    Y[i] = Y[i][0]
performance_standard_deviation = statistics.pstdev(Y)
print("Mean Error: " + str(mean_error))
print("Standard Deviation of Performance: " + str(performance_standard_deviation))


# In[15]:


#Cross Validation

train_size = X.shape[0] * 0.7
train_size = round(train_size)
test_size = X.shape[0] - train_size

#Combine X and Y to shuffle them together, so they still correspond to the same values
Y = numpy.array(Y)
mean_errors = []
for i in range(1000):
    total_data = numpy.zeros([X.shape[0], X.shape[1] + 1])
    total_data[:,0:X.shape[1]] = X
    total_data[:,X.shape[1]] = Y

    numpy.random.shuffle(total_data)
    train_data = total_data[:train_size,:]
    test_data = total_data[train_size:,:]
    X_train = train_data[:,:X.shape[1]]
    Y_train = train_data[:,X.shape[1]]
    X_test = test_data[:,:X.shape[1]]
    Y_test = test_data[:,X.shape[1]]

    lm = linear_model.LinearRegression()
    cross_val_model = lm.fit(X_train,Y_train)
    cross_val_predictions = lm.predict(X_test)    

    abs_errors = []
    for i in range(Y_test.shape[0]):
        error = abs(cross_val_predictions[i] - Y_test[i])
        abs_errors.append(error)

    mean_error = sum(abs_errors)/len(abs_errors)
    mean_errors.append(mean_error)
    standard_deviation_Y_test = statistics.pstdev(Y_test)
    #print("Mean Error: " + str(mean_error))
    #print("Standard Deviation of Y Test Data: " + str(performance_standard_deviation))
    
mean_mean_error = sum(mean_errors)/len(mean_errors)
print("Mean Error with Cross Validation: " + str(mean_mean_error))


# In[31]:


import matplotlib.pyplot as plt
reg_param_values = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]

total_data = numpy.zeros([X.shape[0], X.shape[1] + 1])
total_data[:,0:X.shape[1]] = X
total_data[:,X.shape[1]] = Y

param_list = []
mean_error_list = [] 

for i in range(50):
    numpy.random.shuffle(total_data)
    train_data = total_data[:train_size,:]
    test_data = total_data[train_size:,:]
    X_train = train_data[:,:X.shape[1]]
    Y_train = train_data[:,X.shape[1]]
    X_test = test_data[:,:X.shape[1]]
    Y_test = test_data[:,X.shape[1]]

    for i in range(len(reg_param_values)):
        reg_param = reg_param_values[i]
        param_list.append(reg_param)
        
        lm_reg = linear_model.Ridge(alpha=reg_param, normalize = True)

        lm_reg.fit(X_train,Y_train)

        predictions_reg = lm_reg.predict(X_test)
        #print(predictions)
        predictions_reg = predictions_reg.tolist()

        abs_errors = []
        for i in range(Y_test.shape[0]):
            error = abs(predictions_reg[i] - Y_test[i])
            abs_errors.append(error)

        mean_error = sum(abs_errors)/len(abs_errors)
        mean_error_list.append(mean_error)
        standard_deviation_Y_test = statistics.pstdev(Y_test)
        #print("Mean Error (alpha=" + str(reg_param) + "): " + str(mean_error))
        #print("Standard Deviation of Y Test Data: " + str(performance_standard_deviation))
        
plt.plot(param_list, mean_error_list)
plt.xscale('log')
plt.show()


# In[17]:


import math
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#Regression Neural Network
#Model mostly copied from https://machinelearningmastery.com/regression-tutorial-keras-deep-learning-library-python/

# define base model
#model is neural network with one hidden layer of 10 neurons
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(X.shape[1], input_dim=X.shape[1], kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

#evaluate model
estimator = KerasRegressor(build_fn=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, Y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
mean_error = math.sqrt(-results.mean())
print("Mean Error: " + str(mean_error))
