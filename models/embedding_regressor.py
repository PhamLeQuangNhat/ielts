from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class EmbeddingRegressor:
    def __init__(self, model_type="Ridge", pca_dim=500):
        self.model_type = model_type
        self.pca_dim = pca_dim
        self.scaler = StandardScaler(with_mean=False)
        self.pca = None
        match model_type:
            case "Ridge":
                self.model = Ridge(alpha=1.0)
            case "Rf":
                self.model = RandomForestRegressor(n_estimators=300, random_state=42)
            case "Mlp":
                self.model = MLPRegressor(hidden_layer_sizes=(256,128), max_iter=500, random_state=42)
            case _:
                raise ValueError(f"Model type {model_type} not supported!")

    def fit(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        if X.shape[1] > self.pca_dim:
            self.pca = PCA(n_components=self.pca_dim, random_state=42)
            X_scaled = self.pca.fit_transform(X_scaled)
        self.model.fit(X_scaled, y)
        return self

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        if self.pca:
            X_scaled = self.pca.transform(X_scaled)
        return self.model.predict(X_scaled)
