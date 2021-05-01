####### You may need this section to download the dataset you'll need ####### 
import pandas as pd
import datetime
import pandas_datareader.data as web
from pandas import Series, DataFrame


start = datetime.datetime(2012, 6, 1)
end = datetime.datetime(2020, 6, 1)

df = web.DataReader("GOOGL","yahoo" ,start, end)
df.tail()

####### You may need this section to answer questions about the ARIMA model #######
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import time

# complete this part
Input = ##
#

X = Input
start = time.time()
size = len(X)-14 
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(1,1,2)) #do not change the order
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
end = time.time()
print('This took {} seconds.'.format(end - start))

error = mean_squared_error(test, predictions)
print('Test MSE: %.8f' % error)
# plot
from matplotlib import pyplot as plt
def plot_predicted(predicted_data, true_data):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_title('Prediction vs. Actual ')
    ax.plot(true_data, label='True Data', color='black')
    ax.plot(predicted_data, label='Prediction', color='red')
    plt.legend()
    plt.show()

plot_predicted(predictions, test)
