# Day 04 Tree based models knn feature selection

Trained entire data set on:

- Decision Tree classification
- Random forest
- knn  

and calcualted accuracy

Selected 3 features using random forest importance and trained again.
Compared the accuracies.

## Output

ProductivityScore      0.058471
ManagerSupportScore    0.057015
StressLevel            0.056762
dtype: float64
Top 3 features ['ProductivityScore', 'ManagerSupportScore', 'StressLevel']

Before dropping features
Decision tree accuracy:0.5283333333333333
Random forest accuracy:0.6666666666666666
kNN accuracy(best):0.5966666666666667

After dropping features by random forest importance
Decision Tree accuracy when 3 feature:0.5366666666666666
Random Forest accuracy when 3 feature:0.6283333333333333
kNN when 3 feature:0.5783333333333334
