# [Kaggle Titanic](https://www.kaggle.com/c/titanic)
Got a result of [43rd / 3799](https://www.kaggle.com/jianchengyang/results), on 2016/03/26.

***Accuracy with aggregation: 0.832***

***Accuracy without aggregation (on the best SVM): 0.799***

# Methodology
* Data wrangling: extract features like SibSp, Parch. Fare, Sex, Age, Embarked, Title, Pclass, etc. Fill the values of NA and outliers with some skills.
* Grid search cross validation: on SVM, Random Forest, GBDT, AdaBoost, etc.
* Choose models and emsemble (voting, stacking)
* Aggregation with other models


# Self-build framework: Pipeliner (Wrangler-Modeler-Searcher-Visualizer)
***Note: I can not guarantee all the notebook can fully run in this version (you may test it by yourself, the problems may just be some notebook cell orders).***

1. Wrangler: Data wrangling
1. Modeler: Model builder, sklearn model like
1. Searcher: GridSearchCV on the models
1. Visualizer: To make visualization

In this project, there are just Wrangler and Searcher.

# Folder structure
* data: original source from Kaggle description
* notebook: previous notebook I built
* forum: some file found from Kaggle forum
* result: data file to submit

# License
MIT
