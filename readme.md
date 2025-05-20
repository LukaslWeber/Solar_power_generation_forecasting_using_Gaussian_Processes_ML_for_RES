# Solar Power Forecasting Using Gaussian Processes

- Use a GP to predict solar energy generation based on historical solar power and weather data. At each time point, predict the hourly output for the  next day.
- Analyze the forecast errors for the different forecast horizons using the RMSE and CRPS. 
- Use the last available year of data as your test set.

- Predict the column "Solar_Power" from the file: "Realised_Supply_Germany"

- Bonus Task: Test different feature selection approaches for the weather input.

Which package? GPyTorch -> Uses PyTorch, supports time-series GP models and gives predictive distributions
Why not scikit-learn? Uses exact inference -> O(N^3), so it doesn't work with the amount of available samples and is very slow

Maybe use a deep Gaussian Process


# TODO
-[ ] Provide a reasonable starting point for parameters: <br> model.mean_module.initialize(data=train_y) <br> self.covar_module.initialize_from_data(train_x, train_y)
-[ ] Should we also use the Household PV data from Germany?