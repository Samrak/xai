#importing required libraries
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from keras import backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold

#read the file
fileName = 'BIZIM Historical Data'
df = pd.read_csv(fileName+'.csv')
rowCount = 0
fileSize = df.shape[rowCount] 
trainsetSize = int(fileSize / 1.21)


kfold = KFold(n_splits=10, shuffle=True, random_state=0) 


#setting index as date
df['Date'] = pd.to_datetime(df.Date,format='%b %d, %Y')
df.index = df['Date']

#creating dataframe
data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Price'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Price'][i] = data['Price'][i]

#setting index
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

#creating train and test sets
dataset = new_data.values

for train, valid in kfold.split(dataset):
    print('train: %s, valid: %s' % (dataset[train], dataset[valid]))
    
#train = dataset[0:trainsetSize,:]
#valid = dataset[trainsetSize:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=15, batch_size=32, verbose=2)

train_acc = model.evaluate(x_train, y_train, verbose=0)
print('Train Accuracy: %f' % train_acc)

#predicting 246 values, using past 60 from the train data
inputs = new_data[len(new_data) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

RMS = np.sqrt(np.mean(np.power((valid-closing_price),2)))
print('RMS : %f' % RMS)

trainsetSize = 1162
#for plotting
train = new_data[:trainsetSize]
valid = new_data[trainsetSize:]
valid['Predictions'] = closing_price


look_back = 60
close_data = new_data[-look_back:].values
close_data = close_data.reshape(-1,1)
close_data = scaler.transform(close_data)

def predict(num_prediction):
    prediction_list = close_data[-look_back:]
    
    for _ in range(num_prediction):
        x = prediction_list[-look_back:]
        x = x.reshape((1, look_back, 1))
        out = model.predict(x)
        prediction_list = np.append(prediction_list, out)
    prediction_list = prediction_list[-num_prediction:]
    prediction_list = prediction_list.reshape(-1,1)
    prediction_list = scaler.inverse_transform(prediction_list)
    return prediction_list
    
def predict_dates(num_prediction):
    last_date = df['Date'].values[0]
    prediction_dates = pd.date_range(last_date, periods=num_prediction+1).tolist()
    return prediction_dates[1:num_prediction+1]

num_prediction = 60
forecast = predict(num_prediction)
forecast_dates = predict_dates(num_prediction)

##print(forecast_dates)
##print(forecast)

plt.title(fileName, size=10 )
plt.xlabel('Years')
plt.ylabel('Share Value')
plt.plot(train['Price'],label='Price')
plt.plot(valid[['Price','Predictions']])
plt.plot(forecast_dates, forecast)
## Rotate date labels automatically
plt.gcf().autofmt_xdate()
plt.show()

#yhat_classes = np.argmax(valid['Predictions'], axis=-1)

# predict crisp classes for test set

actual = valid['Price'].astype('float64') 
actual = np.asarray(actual)
pred = valid['Predictions'].round(2);
pred = np.asarray(pred)

def evaluate(rawData):
    evaluated = np.zeros(len(rawData)-1)
    i = 1
    while i < len(actual):
        evaluated[i-1] = 0
        if actual[i-1] > actual[i]:
            evaluated[i-1] = -1
        elif actual[i-1] < actual[i]:
            evaluated[i-1] = 1  
        # print('i = %f, i+1 = %f res = %f' % (rawData[i-1], rawData[i], evaluated[i-1]))
        i += 1   
    return evaluated

def getPerformanceMetrics(actual, prediction):
    tp = 0 
    tn = 0 
    fp = 0 
    fn = 0
    i = 0
    while i < len(actual):
        if actual[i] == prediction[i]:
            if actual[i] > 0:
                tp+=1   
            else:
               tn+=1   
        elif prediction[i] > actual[i]:
            fp+=1;
        elif prediction[i] < actual[i]:
            fn+=1;
        i += 1
        
    print('truePositive = %f, trueNegative = %f' % (tp, tn))
    print('falsePositive = %f, falseNegative = %f' % (fp, fn))
    return tp, tn, fp, fn
  
evaluated_actual = evaluate(actual)
evaluated_pred = evaluate(pred)

tp, tn, fp, fn = getPerformanceMetrics(evaluated_actual,evaluated_pred)    

# accuracy: (tp + tn) / (tp + tn + fp + fn)
accuracy = (tp + tn) / (tp + tn + fp + fn)
print('Accuracy: %f' % accuracy)
# precision tp / (tp + fp)
precision = tp / (tp + fp)
print('Precision: %f' % precision)
# recall: tp / (tp + fn)
recall = tp / (tp + fn)
print('Recall: %f' % recall)
# f1: 2*tp / (2*tp + fp + fn)
f1 = 2*tp / (2*tp + fp + fn)
print('F1 score: %f' % f1)
