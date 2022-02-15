import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn

#set parameters
plot_params = dict(
    color = '0.75',
    style = '.-',
    markeredgecolor = '0.25',
    markerfacecolor='0.25',
    legend = False
)

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

#core script

tunnel = pd.read_csv('./tunnel.csv', parse_dates=['Day'])
tunnel = tunnel.set_index('Day')
tunn = tunnel.to_period()
print(tunnel.head())
len(tunnel)

tunnel['Time'] = np.arange(len(tunnel.index))
tunnel['lag'] = tunnel['NumVehicles'].shift(1)
tunnel.dropna(inplace = True)
print(tunnel.head())

experiment_id = mlflow.set_experiment('cars-through-tunnel').experiment_id

#determine where to truncate the training set
test_perc = .25
test_size= round(test_perc * len(tunnel))

# Split the data into training and test sets. (1 - test_perc, test_perc) split.
test = tunnel.iloc[-test_size:, :]
train = tunnel.iloc[:-test_size:, :]

# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["NumVehicles"], axis=1)
test_x = test.drop(["NumVehicles"], axis=1)
train_y = train[["NumVehicles"]]
test_y = test[["NumVehicles"]]

with mlflow.start_run():
    model = LinearRegression()
    model.fit(train_x, train_y)
 
    predicted_qualities = model.predict(test_x)
    
    y_pred = pd.Series(predicted_qualities.flatten(), index = test_x.index)
    
    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    print("Linear Regression model:")
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)
    print("test_perc: %s" % test_perc)

    #fig, ax = plt.subplots()
    ax = test_y.plot(**plot_params)
    ax=y_pred.plot(ax=ax, linewidth = 3, figsize = (15, 8))
    ax.set_title('Time Plot of Tunnel Traffic')

    regress_pic = plt.savefig('ts-regress', ax = ax)

    tags = {'purpose' : 'education',
            'problem' : 'time-series'}

    mlflow.log_param('test_perc', test_perc)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    mlflow.log_artifact("ts-regress.png")
    mlflow.set_tags(tags)

    tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme


    # Model registry does not work with file store
    if tracking_url_type_store != "file":

        # Register the model
        # There are other ways to use the Model Registry, which depends on the use case,
        # please refer to the doc for more information:
        # https://mlflow.org/docs/latest/model-registry.html#api-workflow
        mlflow.sklearn.log_model(model, "model", registered_model_name="LinRegTunnelModel")
    else:
        mlflow.sklearn.log_model(model, "model")

    #ends the run
    mlflow.end_run()