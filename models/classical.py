from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from scipy.sparse import issparse

class ClassicalClassifier:
    def __init__(self, model_type="RandomForest"):
        self.imputer = SimpleImputer(strategy="mean")
        match model_type:
            case 'RandomForest':
                self.model = RandomForestClassifier(n_estimators=300, random_state=42)
            case 'XGBoost':
                self.model = XGBClassifier( 
                    n_estimators=200, max_depth=6, random_state=42,
                    eval_metric='mlogloss', use_label_encoder=False
                )
            case 'LogisticRegression':
                self.model = LogisticRegression(    
                    max_iter=1000, multi_class='multinomial', solver='lbfgs'
                )       
            case 'RidgeClassifier':
                self.model = RidgeClassifier()
            case _:
                raise ValueError(f"Model type {model_type} not supported!")

    def _fit_transform_X(self, X):
        X_dense = X.toarray() if issparse(X) else X
        return self.imputer.fit_transform(X_dense)

    def _transform_X(self, X):
        X_dense = X.toarray() if issparse(X) else X
        return self.imputer.transform(X_dense)

    def fit(self, X, y):
        X_prepared = self._fit_transform_X(X)
        self.model.fit(X_prepared, y)
        return self

    def predict(self, X):
        X_prepared = self._transform_X(X)
        return self.model.predict(X_prepared)



