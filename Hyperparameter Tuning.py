
# Implementation of Grid search

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
grid_vals = {'penalty': ['l1','l2'], 'C': [0.001,0.01,0.1,1]}
grid_lr = GridSearchCV(estimator=model, param_grid=grid_vals, scoring='accuracy', 
                       cv=6, refit=True, return_train_score=True) 

# Training and Prediction
grid_lr.fit(X_train, y_train)
preds = grid_lr.best_estimator_.predict(X_test)


# Implementation of Random Search
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

model = RandomForestClassifier()

param_vals = {'max_depth': [200,500,800,1100], 'n_estimators': [100,200,300,400], 'learning_rate': [0.001,0.01,0.1,1,10]}
random_rf = RandomizedSearchCV(estimator=model, param_distributions= param_vals, n_iter=10, scoring='accuracy', cv=5, refit= True, n_jobs=-1)
X= param_vals['max_depth']
Y= param_vals['n_estimators']

#Training and Prediction
X_train, Y_train, X_test, Y_test = train_test_split(X, Y, train_size=0.7, random_state=0)
random_rf.fit(X_train, Y_train)
preds = random_rf.best_estimator_.predict(X_test)

# Genetic Algorithm in Python
from tpot import TPOTClassifier
tpot_clf = TPOTClassifier(generations=100, population_size=100, verbosity=2, offspring_size=100, scoring='accuracy', cv=6)

# Training and prediction
tpot_clf.fit(X_train, Y_test)
