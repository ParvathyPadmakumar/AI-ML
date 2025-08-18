# Day 02:Feature Forge-Regression

## Documentation

Encoding nominal data using One-Hot Encoding
Ordinal data using OrdinalEncoder
Numeral data scaled using min-max scaling

Regression using mutual_info_regression and correlation with StressLevel
Two interaction features created:

- Stressxworkhours
- Sleep/Stress
Three regression models trained:
- modellinear
- modelridge
- modellasso
MSE and R2 score calculated for each model

Results:
Linear Regression MSE= 0.0028657547747736514
Linear Regression R2= 0.964948414032654
Ridge Regression MSE= 0.0028602646764333485
Ridge Regression R2= 0.9650155644586563
Lasso Regression MSE= 0.013145132708199184
Lasso Regression R2= 0.8392194080144182

```text
MSE when lower is better
R2 when closer to 1 is better
### Hence, Ridge Regression performed best  
```

Explanation:
MSE measures the average squared difference between actual and predicted values.Lower MSE means the model's predictions are closer to the true values.
R2 indicates how well the model explains the variance in the target variable.R2 ranges from 0 to 1,sometimes negative if model is very poor.R² next to 1 means the model explains more variance spread.
