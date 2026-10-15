"""
regulatory_classifier.py - ML Engine M2: IRA Section 30D Compliance Classifier & 45X Calculator
Author: Bhupesh Chandra Dimri
Target: GM Regulatory & Strategy Compliance Modeling
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

class RegulatoryComplianceEngine:
    """
    ML Engine M2:
    Classifies vehicle program compliance under IRS Section 30D (Clean Vehicle Credit)
    and computes advanced manufacturing production tax credits under Section 45X.
    """
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.classifier = RandomForestClassifier(
            n_estimators=120, max_depth=5, random_state=random_state
        )
        self.feature_columns = []
        self.accuracy = 0.0

    def train(self, df_features: pd.DataFrame) -> Dict[str, Any]:
        regulatory_features = [
            "mineral_fta_sourcing_pct", "battery_na_assembly_pct",
            "feoc_breach_flag", "regulatory_compliance_composite",
            "battery_kwh", "cell_cost_per_kwh"
        ]
        self.feature_columns = regulatory_features
        
        X = df_features[self.feature_columns]
        y = df_features["tax_credit_eligible"] # 0 = $0, 1 = $3,750, 2 = $7,500
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=self.random_state, stratify=y
        )
        
        self.classifier.fit(X_train, y_train)
        preds = self.classifier.predict(X_test)
        
        self.accuracy = float(accuracy_score(y_test, preds))
        report = classification_report(y_test, preds, output_dict=True)
        
        return {
            "accuracy": self.accuracy,
            "classification_report": report
        }

    def predict_credit(self, df_features: pd.DataFrame) -> np.ndarray:
        X = df_features[self.feature_columns]
        predicted_classes = self.classifier.predict(X)
        
        # Map class back to dollar amounts: 0 -> $0, 1 -> $3,750, 2 -> $7,500
        credit_map = {0: 0.0, 1: 3750.0, 2: 7500.0}
        return np.vectorize(credit_map.get)(predicted_classes)

    @staticmethod
    def calculate_section_45x_offset(battery_kwh: float, cell_credit: float = 35.0, module_credit: float = 10.0) -> float:
        """
        Computes Section 45X Advanced Manufacturing Credit ($/unit) offsetting COGS.
        """
        total_credit_per_kwh = cell_credit + module_credit # $45/kWh
        return float(battery_kwh * total_credit_per_kwh)
