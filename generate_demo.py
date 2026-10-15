"""
generate_demo.py - End-to-End Pipeline Execution Script
ml-project-gm-vehicle-portfolio-optimizer
Author: Bhupesh Chandra Dimri | Target: General Motors Senior Finance Analyst
"""

import os
import sys
import pandas as pd

from src.data_pipeline import generate_vehicle_program_data, extract_features
from src.cost_prediction_engine import VehicleCostPredictionEngine
from src.regulatory_classifier import RegulatoryComplianceEngine
from src.portfolio_dcf_engine import PortfolioDCFEngine

def run_portfolio_demo():
    print("=" * 85)
    print("STARTING: ml-project-gm-vehicle-portfolio-optimizer")
    print("Candidate: Bhupesh Chandra Dimri | Target: General Motors Senior Finance Analyst")
    print("Focus: Corporate Strategy, Portfolio Planning & Vehicle Line Financial Governance")
    print("=" * 85)

    # 1. Ingestion of Multi-Stream Feeds (D1, D2, D3)
    print("\n[Step 1] Ingesting multi-source automotive feeds (D1: GLPT, D2: eGPPS, D3: Macro/IRA)...")
    df_raw = generate_vehicle_program_data(n_samples=1200)
    print(f"-> Successfully loaded {len(df_raw)} vehicle platform program lines across 4 assembly plants.")
    print(df_raw[["program_id", "platform", "trim", "battery_kwh", "lifetime_volume", "base_msrp"]].head(3).to_string())

    # 2. Feature Engineering
    print("\n[Step 2] Engineering automotive features (battery density, tooling per unit, ECO ratios)...")
    df_feat = extract_features(df_raw)
    print(f"-> Generated {df_feat.shape[1]} engineered features for machine learning pipelines.")

    # 3. Train ML Engine M1 (BOM & Tooling Overrun Predictor)
    print("\n[Step 3] Training ML Engine M1: Gradient Boosting Cost & CapEx Overrun Regressors...")
    cost_engine = VehicleCostPredictionEngine()
    cost_metrics = cost_engine.train(df_feat)
    print(f"-> BOM Cost Model Trained: RMSE = ${cost_metrics['bom_rmse']:,.2f} | MAPE = {cost_metrics['bom_mape_pct']:.2f}% | R² = {cost_metrics['bom_r2']:.4f}")
    print(f"-> Tooling CapEx Model:   RMSE = ${cost_metrics['tooling_rmse']:,.2f} | MAPE = {cost_metrics['tooling_mape_pct']:.2f}% | R² = {cost_metrics['tooling_r2']:.4f}")
    
    # Feature Importances
    df_imp = cost_engine.get_feature_importances(top_n=5)
    print("\nTop 5 Cost Driver Importances (M1):")
    print(df_imp.to_string(index=False))

    # 4. Train ML Engine M2 (Regulatory Compliance Classifier)
    print("\n[Step 4] Training ML Engine M2: IRA Section 30D / 45X Compliance & Credit Classifier...")
    reg_engine = RegulatoryComplianceEngine()
    reg_metrics = reg_engine.train(df_feat)
    print(f"-> Section 30D Sourcing Classifier Accuracy: {reg_metrics['accuracy'] * 100:.2f}%")
    
    # 5. Generate Deliverable O1 (Program Gateway Review DCF)
    print("\n[Step 5] Generating Automated Deliverable O1: Program Gateway Review DCF (GBCT)...")
    dcf_engine = PortfolioDCFEngine(discount_rate=0.10)
    
    # Simulate Next-Gen Midsize Electric CUV Program
    o1_dcf = dcf_engine.generate_program_gateway_dcf(
        program_name="Project Titan - Next-Gen Ultium Midsize CUV",
        annual_volume=85000,
        lifecycle_years=7,
        msrp=54000.0,
        unit_bom=32500.0,
        tooling_capex=520e6,
        battery_kwh=85.0,
        sec_45x_offset=85.0 * 45.0 # $3,825 credit
    )
    
    print(f"Program: {o1_dcf['program_name']}")
    print(f"Lifetime Volume: {o1_dcf['lifetime_units']:,} units over 7 years")
    print(f"Dedicated Tooling Amortization (UOP): ${o1_dcf['tooling_amort_per_unit']:,.2f}/unit")
    print(f"Net Realized COGS (incl. Sec 45X Offset): ${o1_dcf['net_unit_cogs']:,.2f}/unit")
    print(f"Unit Contribution Margin: ${o1_dcf['unit_contribution_margin']:,.2f}/unit ({o1_dcf['margin_pct']:.2f}%)")
    print(f"Annual EBITDA: ${o1_dcf['annual_ebitda']/1e6:,.2f}M | Lifetime EBITDA: ${o1_dcf['lifetime_ebitda']/1e6:,.2f}M")
    print(f"Platform NPV @ 10%: ${o1_dcf['npv_10pct']/1e6:,.2f}M | IRR: {o1_dcf['irr_pct']:.2f}% | Payback: {o1_dcf['discounted_payback_years']:.2f} yrs")
    print(f"Executive Gateway Status: {o1_dcf['gateway_recommendation']}")

    # 6. Generate Deliverable O2 (Corporate Strategy 8-Level Variance Bridge)
    print("\n[Step 6] Generating Automated Deliverable O2: Corporate Strategy 8-Level Variance Bridge...")
    o2_bridge = dcf_engine.generate_8level_variance_bridge(
        baseline_units=750000,
        actual_units=600000,
        baseline_msrp=52000.0,
        actual_msrp=53500.0,
        baseline_bom=31000.0,
        actual_bom=33450.0,
        baseline_tooling=450e6,
        actual_tooling=565e6,
        baseline_45x=3825.0,
        actual_45x=3825.0
    )
    
    print(f"Baseline Lifetime EBITDA: ${o2_bridge['baseline_ebitda']/1e6:,.2f}M")
    print(f"Actual Realized EBITDA:   ${o2_bridge['actual_ebitda']/1e6:,.2f}M")
    print(f"Net Variance Bridge:      ${o2_bridge['total_variance']/1e6:,.2f}M\n")
    print("Waterfall Bridge Decomposition:")
    for step in o2_bridge['waterfall_steps']:
        print(f"  - {step['label']:<32} : ${step['value']/1e6:+10,.2f}M")

    print("\n" + "=" * 85)
    print("SUCCESS: ml-project-gm-vehicle-portfolio-optimizer demonstration finished.")
    print("=" * 85)

if __name__ == "__main__":
    run_portfolio_demo()
