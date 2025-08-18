# Tweet-sentiment-analysis

## Documentation

Removed unnecessary parts of each tweet
Converted labels into -1,0,1
Used TF-IDF for feature importance

## TF-IDF for feature importance

tf-idf helps analyze which words have most importance in the tweet and grade them accordingly. Hence, the features with most importance can be selcted according to their score.

## Model selection

- high-dimensional data(SVM possible)
- data trained with tf-idf so, matrix has sparse linear values.
- sentiment only has 3 values so Logistic Regression is possible

- Implemented cross validation to compare models
- Logistic Regression was most accurate and fast enough.

## Challenges faced and how you solved them

1. Model selection was hard and took some time

Problems encountered and solutions:
1.1  

- model4=XGBClassifier(random_state=42)
score_xgb=cross_val_score(model4,x,y,cv=5,scoring='accuracy')
print("cross validation accuracy:",score_xgb.mean())

- Here, y should be either 0 or 1. Here, y can be -1,0 or 1.
- Hence converting -1 to 0 here for model training.

1.2  

- model2=SVC(kernel='linear',C=0.1,gamma='scale',random_state=42)
score_svm=cross_val_score(model2,x,y,cv=5,scoring='accuracy')
print("cross validation accuracy linear svm:",score_svm.mean())

- This was too slow on this large dataset

2.  I labelled the columns for easiness during training.

- df.columns=['sentiment','id','date','ifquery','username','tweetcontent']

3.  Sentiment labelling

- I labelled sentiments as string first but that can not be used for vectorization here. So i changed to literals for easier training.

Changed: map_strlabels={0:'Negative',2:'Neutral',4:'Positive'}
Fix: map_numlabels={0:-1,2:0,4:1}
