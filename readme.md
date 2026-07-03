# Solar Power Forecasting Using Gaussian Processes

![img.png](GP_pred_image.png)

In this project, I'll try to predict German's solar power generation with Gaussian Processes. 
It was created for a university seminar (Machine Learning for Renewable Energy Systems).

---
### The task:
- Use a GP to predict solar energy generation based on historical solar power and weather data. At each time point, predict the hourly output for the  next day.
- Analyze the forecast errors for the different forecast horizons using the RMSE and CRPS. 
---
This project uses GPyTorch, which itself uses PyTorch under the hood.

The whoel data analysis, model definitions end everything else done during this project can be found in `GP_Solar_Prediction.ipynb`.
