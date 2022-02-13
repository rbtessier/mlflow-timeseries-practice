import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

tunnel = pd.read_csv('./tunnel.csv', parse_dates=['Day'])
print(tunnel.head())