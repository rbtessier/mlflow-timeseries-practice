# MLFlow practice

This is a project to help me(mR. T) get familiar with MLFlow. Previously I used the experiment tracking in Azure ML studio but the commands are specific to that platform. At work I don't have an Azure account and the organization seems sticky with security so this should be kept separate from my work ML toolkit. 

I'm also using this as a way to build up my ml toolkit from my personal computer - the reusable pieces will be implemented in my ML models at work. I'm hoping to build out a toolkit that allows for rapid model training, testing, tracking, and deployment.

I'm going to be using the Kaggle time series forcasting data found here https://www.kaggle.com/c/store-sales-time-series-forecasting. This is relevant to my work data since we want to do a lot of forecasting over time.

## Large Files in Github

To work with files over 50 mb with git, I needed to use git LFS for the files that are too large, info on installationto be found here https://git-lfs.github.com/

### Feb 13

This weekend I am trying out code I found in the Kaggle Time Series tutorial for running linear regressions on time series data that model *serial dependence* using a `lag` feature and assign an ordering to the data

### Feb 14

Yesterday I learned a lot - I spent a ton of time just getting my figure to plot correctly and upload to MLFlow. Further I realized I was being dumb with my train-test split - I was splitting randomly instead of at a specific time.