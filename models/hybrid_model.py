# models/hybrid_model.py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, RegressorMixin

# Stacking hybrid regressor
class HybridStackingRegressor(BaseEstimator, RegressorMixin):
    """
    Stack two models:
    - handcrafted features -> model1
    - embedding features -> model2
    Meta-learner combines outputs to final prediction
    """
    def __init__(self, model_handcrafted=None, model_embed=None, meta_learner=None):
        self.model_handcrafted = model_handcrafted if model_handcrafted else RandomForestRegressor(n_estimators=200, random_state=42)
        self.model_embed = model_embed if model_embed else Ridge(alpha=1.0)
        self.meta_learner = meta_learner if meta_learner else Ridge(alpha=1.0)

    def fit(self, X_hc, X_emb, y):
        self.scaler_hc = StandardScaler()
        self.scaler_emb = StandardScaler()
        
        X_hc_scaled = self.scaler_hc.fit_transform(X_hc)
        X_emb_scaled = self.scaler_emb.fit_transform(X_emb)

        # Optional PCA to reduce dimension of embeddings
        if X_emb_scaled.shape[1] > 500:  
            self.pca = PCA(n_components=500, random_state=42)
            X_emb_scaled = self.pca.fit_transform(X_emb_scaled)
        else:
            self.pca = None

        # Fit base models
        self.model_handcrafted.fit(X_hc_scaled, y)
        self.model_embed.fit(X_emb_scaled, y)

        # Meta-learner input: predictions of base models
        pred_hc = self.model_handcrafted.predict(X_hc_scaled).reshape(-1,1)
        pred_emb = self.model_embed.predict(X_emb_scaled).reshape(-1,1)
        meta_input = np.hstack([pred_hc, pred_emb])
        self.meta_learner.fit(meta_input, y)
        return self

    def predict(self, X_hc, X_emb):
        X_hc_scaled = self.scaler_hc.transform(X_hc)
        X_emb_scaled = self.scaler_emb.transform(X_emb)
        if self.pca:
            X_emb_scaled = self.pca.transform(X_emb_scaled)
        
        pred_hc = self.model_handcrafted.predict(X_hc_scaled).reshape(-1,1)
        pred_emb = self.model_embed.predict(X_emb_scaled).reshape(-1,1)
        meta_input = np.hstack([pred_hc, pred_emb])
        return self.meta_learner.predict(meta_input)