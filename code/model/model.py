from extract_features import create_feature_results_matrix
from sklearn.externals import joblib
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import RandomizedPCA
from sklearn.cross_validation import cross_val_score

datapath = '/home/ankesh/masters-thesis/data/reviews_Cell_Phones_and_Accessories.json.gz'

#X, y = create_feature_results_matrix(datapath)
#print 'Dumping matrices to disk.'
filename_X = 'X.joblib.pkl'
filename_y = 'y.joblib.pkl'
#_ = joblib.dump(X, filename_X, compress=9)
#_ = joblib.dump(y, filename_y, compress=9)
print 'Loading matrices'
X = joblib.load(filename_X)
y = joblib.load(filename_y)

X = X.toarray()
print X.shape

from sklearn import svm
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition, pipeline, metrics, grid_search

print 'Truncated SVD'
pca = RandomizedPCA(n_components=200)
X_reduced = pca.fit_transform(X)

print 'Standard Scaler'
scl = StandardScaler(with_mean=False)
X_scaled = scl.fit_transform(X_reduced)

#svm_model = LinearSVR()
#clf = pipeline.Pipeline([('svd', svd),('scl', scl),('svm', svm_model)])
#param_grid = {'svm__C': [1,3]}
#print 'Grid Search started'
#model_svm = grid_search.GridSearchCV(estimator = clf, param_grid=param_grid, scoring='mean_squared_error',n_jobs=-1, iid=True, refit=True, cv=10)
#model_svm.fit(X,y)
#print model_svm.best_score_
#print model_svm.best_estimator_

from sklearn.linear_model.stochastic_gradient import SGDRegressor
sgd = SGDRegressor()

print 'Cross Validation started'
scores = cross_val_score(sgd, X_scaled, y, cv=2, scoring='mean_squared_error')
print scores
print scores.mean()
