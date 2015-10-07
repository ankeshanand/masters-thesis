from extract_features import create_feature_results_matrix
from sklearn.externals import joblib

datapath = '/home/ankesh/masters-thesis/data/reviews_Cell_Phones_and_Accessories.json.gz'

X, y = create_feature_results_matrix(datapath)
print 'Dumping matrices to disk.'
filename_X = 'X.joblib.pkl'
filename_y = 'y.joblib.pkl'
_ = joblib.dump(X, filename_X, compress=9)
_ = joblib.dump(y, filename_y, compress=9)

from sklearn import svm
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search

scl = StandardScaler(with_mean=False)
svm_model = LinearSVR()
clf = pipeline.Pipeline([('scl', scl),('svm', svm_model)])
param_grid = {'svm__C': [1,3]}
print 'Grid Search started'
model_svm = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring='mean_squared_error',n_jobs=-1, iid=True, refit=True, cv=10)
model_svm.fit(X,y)
print model_svm.best_score_
print model_svm.best_estimator_
