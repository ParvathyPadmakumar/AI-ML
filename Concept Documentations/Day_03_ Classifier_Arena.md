# Day 03:Feature ClassifierArena

## Documentation

- Trained one model on only LogicalRegression and another on EDA+LogicalRegression
- Was facing some errors without scaling, so scaled the x values before model training

- Results:

Logistic Regression Trained with LDA
Accuracy of model is:0.9833333333333333
Confusion Matrix:[[392   7]
 [  3 198]]

 Logistic Regression Trained without LDA
Accuracy of model is:0.99
Confusion Matrix:[[395   4]
 [  2 199]]

- Hence, the model without EDA performed better in my analysis. It had more accuracy, more true positives and negatives and less false positives and negatives.

- Plotted AUC and ROC

### Logistic Regression

- Binary classification into Yes/No.

### Points in EDA

- EDA maximises distance between means and reduces scatter to make a new axis
- The axes is compared based on variance
- It reduces the dimension  

### ROC

- True positive rate(y) vs false positive rate(x)  
