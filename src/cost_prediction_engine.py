"""
cost_prediction_engine.py - Machine Learning Engine M1: Vehicle BOM & Tooling Overrun Predictor
Author: Bhupesh Chandra Dimri
Target: GM Program Finance Gateway Reviews
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

class VehicleCostPredictionEngine:
    """
    ML Engine M1:
    Predicts final realized vehicle Bill of Materials (BOM) piece cost and dedicated
    tooling CapEx expenditure from early gateway baseline targets.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.bom_model = GradientBoostingRegressor(
            n_estimators=180, learning_rate=0.06, max_depth=4, random_state=random_state
        )
        self.tooling_model = RandomForestRegressor(
            n_estimators=150, max_depth=6, random_state=random_state
        )
        self.feature_columns = []
        self.metrics = {}

    def train(self, df_features: pd.DataFrame) -> Dict[str, float]:
        exclude_cols = [
            "program_id", "realized_unit_bom", "realized_tooling_capex",
            "tax_credit_eligible", "tax_credit_amount", "sec_45x_unit_credit"
        ]
        self.feature_columns = [col for col in df_features.columns if col not in exclude_cols]
        
        X = df_features[self.feature_columns]
        y_bom = df_features["realized_unit_bom"]
        y_tooling = df_features["realized_tooling_capex"]
        
        X_train, X_test, y_bom_train, y_bom_test, y_tool_train, y_tool_test = train_test_split(
            X, y_bom, y_tooling, test_size=0.20, random_state=self.random_state
        )
        
        # Fit BOM Regressor
        self.bom_model.fit(X_train, y_bom_train)
        bom_preds = self.bom_model.predict(X_test)
        
        # Fit Tooling Regressor
        self.tooling_model.fit(X_train, y_tool_train)
        tool_preds = self.tooling_model.predict(X_test)
        
        self.metrics = {
            "bom_rmse": float(np.sqrt(mean_squared_error(y_bom_test, bom_preds))),
            "bom_mape_pct": float(mean_absolute_percentage_error(y_bom_test, bom_preds) * 100),
            "bom_r2": float(r2_score(y_bom_test, bom_preds)),
            "tooling_rmse": float(np.sqrt(mean_squared_error(y_tool_test, tool_preds))),
            "tooling_mape_pct": float(mean_absolute_percentage_error(y_tool_test, tool_preds) * 100),
            "tooling_r2": float(r2_score(y_tool_test, tool_preds))
        }
        return self.metrics

    def predict(self, df_features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = df_features[self.feature_columns]
        pred_bom = self.bom_model.predict(X)
        pred_tooling = self.tooling_model.predict(X)
        return pred_bom, pred_tooling
        
    def get_feature_importances(self, top_n: int = 6) -> pd.DataFrame:
        importances = self.bom_model.feature_importances_
        df_imp = pd.DataFrame({
            "Feature": self.feature_columns,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).head(top_n)
        return df_imp
