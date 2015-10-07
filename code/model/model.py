from extract_features import create_feature_results_matrix

datapath = '/home/ankesh/masters-thesis/data/reviews_Cell_Phones_and_Accessories.json.gz'

X, y = create_feature_results_matrix(datapath)

from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search

scl = StandardScaler()
svm_model = svm.SVR()
clf = pipeline.Pipeline([('scl', scl),('svm', svm_model)])
param_grid = {'svm__C': [i+1 for i in xrange(5)], 'svm__kernel': ['linear', 'rbf']}
model_svm = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring='mean_squared_error',n_jobs=-1, iid=True, refit=True, cv=10)
model_svm.fit(X,y)
print model_svm.best_score_
