"""
data_pipeline.py - Synthetic Automotive Program Data Generator & Feature Engineering
Author: Bhupesh Chandra Dimri
Target: GM Vehicle Program Finance & Portfolio Strategy
"""

import numpy as np
import pandas as pd

def generate_vehicle_program_data(n_samples: int = 1200, random_state: int = 42) -> pd.DataFrame:
    """
    Simulates enterprise vehicle program data mirroring GM systems:
    - D1 (GLPT): Production volume profiles, vehicle platforms, and trim distribution.
    - D2 (eGPPS): Bill of Materials (BOM) piece targets, dedicated tooling CapEx, engineering change orders (ECOs).
    - D3 (Macro/Policy): Battery cell $/kWh, lithium price index, IRA 30D sourcing ratios, and FEOC flags.
    """
    np.random.seed(random_state)
    
    platforms = ["BEV3_Midsize_CUV", "BEV3_Fullsize_SUV", "BT1_Electric_Truck", "Global_Compact_EV"]
    trims = ["LT_Standard", "RS_Sport", "Premier_Luxury", "High_Country_Flagship"]
    assembly_plants = ["Spring_Hill_TN", "Factory_ZERO_Detroit", "Orion_Assembly_MI", "Ramos_Arizpe_MX"]
    
    data = []
    
    for i in range(n_samples):
        program_id = f"GM-PRG-{1000 + i}"
        platform = np.random.choice(platforms, p=[0.40, 0.25, 0.20, 0.15])
        trim = np.random.choice(trims, p=[0.35, 0.30, 0.20, 0.15])
        plant = np.random.choice(assembly_plants)
        
        # D1: GLPT Volume Data
        lifecycle_years = 7
        annual_volume = np.random.randint(45000, 160000)
        lifetime_volume = annual_volume * lifecycle_years
        
        # Vehicle Specs based on platform
        if platform == "BT1_Electric_Truck":
            battery_kwh = np.random.choice([150, 180, 205])
            base_msrp = np.random.uniform(72000, 98000)
            baseline_bom = np.random.uniform(46000, 58000)
            tooling_capex = np.random.uniform(650e6, 950e6)
        elif platform == "BEV3_Fullsize_SUV":
            battery_kwh = np.random.choice([100, 120, 140])
            base_msrp = np.random.uniform(65000, 84000)
            baseline_bom = np.random.uniform(39000, 48000)
            tooling_capex = np.random.uniform(500e6, 750e6)
        elif platform == "BEV3_Midsize_CUV":
            battery_kwh = np.random.choice([75, 85, 100])
            base_msrp = np.random.uniform(48000, 64000)
            baseline_bom = np.random.uniform(29000, 38000)
            tooling_capex = np.random.uniform(400e6, 620e6)
        else: # Global Compact EV
            battery_kwh = np.random.choice([55, 65])
            base_msrp = np.random.uniform(34000, 44000)
            baseline_bom = np.random.uniform(22000, 29000)
            tooling_capex = np.random.uniform(300e6, 450e6)
            
        # D2: eGPPS BOM Details & Complexity
        num_tier1_suppliers = np.random.randint(65, 140)
        num_engineering_change_orders = np.random.poisson(lam=18)
        chassis_weight_kg = np.random.normal(2100, 250)
        
        # D3: Macro, Commodity & Policy Feeds
        cell_cost_per_kwh = np.random.normal(108, 14)  # $108/kWh mean
        lithium_carbonate_index = np.random.normal(100, 15)  # baseline 100 index
        
        # IRA Section 30D Qualifying Ratios
        mineral_fta_sourcing_pct = np.clip(np.random.normal(0.68, 0.12), 0.30, 0.95)
        battery_na_assembly_pct = np.clip(np.random.normal(0.76, 0.10), 0.40, 0.98)
        feoc_breach_flag = np.random.choice([0, 1], p=[0.92, 0.08]) # 8% risk of FEOC contaminant
        
        # Target Variables (True Cost Deviations & Final Realized Metrics)
        # 1. Realized Unit BOM Cost (Driven by cell price, lithium spikes, ECO volume)
        bom_inflation_factor = (
            1.0 
            + (cell_cost_per_kwh - 100) * 0.003 
            + (lithium_carbonate_index - 100) * 0.0015 
            + (num_engineering_change_orders * 0.004)
            + np.random.normal(0, 0.02)
        )
        realized_unit_bom = baseline_bom * bom_inflation_factor
        
        # 2. Realized Tooling CapEx Overrun
        tooling_overrun_pct = np.clip(
            (num_engineering_change_orders * 0.012) + np.random.normal(0.04, 0.03), 
            -0.05, 0.35
        )
        realized_tooling_capex = tooling_capex * (1.0 + tooling_overrun_pct)
        
        # 3. Regulatory Eligibility Ground Truth (Section 30D)
        # 2025/2026 standard: Minerals >= 60%, Assembly >= 70%, FEOC == 0
        min_eligible = 1 if (mineral_fta_sourcing_pct >= 0.60 and feoc_breach_flag == 0) else 0
        comp_eligible = 1 if (battery_na_assembly_pct >= 0.70 and feoc_breach_flag == 0) else 0
        
        if feoc_breach_flag == 1:
            tax_credit_eligible = 0
            tax_credit_amount = 0.0
        else:
            tax_credit_amount = (min_eligible * 3750.0) + (comp_eligible * 3750.0)
            if tax_credit_amount == 7500.0:
                tax_credit_eligible = 2 # Full Credit
            elif tax_credit_amount == 3750.0:
                tax_credit_eligible = 1 # Half Credit
            else:
                tax_credit_eligible = 0 # No Credit
                
        # 4. Section 45X Advanced Manufacturing Credit ($45/kWh: $35 cell + $10 module)
        sec_45x_unit_credit = battery_kwh * 45.0
        
        data.append({
            "program_id": program_id,
            "platform": platform,
            "trim": trim,
            "assembly_plant": plant,
            "lifetime_volume": lifetime_volume,
            "annual_volume": annual_volume,
            "battery_kwh": battery_kwh,
            "base_msrp": base_msrp,
            "baseline_bom": baseline_bom,
            "tooling_capex_target": tooling_capex,
            "num_tier1_suppliers": num_tier1_suppliers,
            "num_engineering_change_orders": num_engineering_change_orders,
            "chassis_weight_kg": chassis_weight_kg,
            "cell_cost_per_kwh": cell_cost_per_kwh,
            "lithium_carbonate_index": lithium_carbonate_index,
            "mineral_fta_sourcing_pct": mineral_fta_sourcing_pct,
            "battery_na_assembly_pct": battery_na_assembly_pct,
            "feoc_breach_flag": feoc_breach_flag,
            # Targets
            "realized_unit_bom": realized_unit_bom,
            "realized_tooling_capex": realized_tooling_capex,
            "tax_credit_eligible": tax_credit_eligible,
            "tax_credit_amount": tax_credit_amount,
            "sec_45x_unit_credit": sec_45x_unit_credit
        })
        
    return pd.DataFrame(data)

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms raw automotive engineering and purchasing feeds into ML features.
    """
    df_feat = df.copy()
    
    # Financial Ratios & Engineering Proxies
    df_feat["battery_kwh_per_ton"] = df_feat["battery_kwh"] / (df_feat["chassis_weight_kg"] / 1000.0)
    df_feat["tooling_per_unit_baseline"] = df_feat["tooling_capex_target"] / df_feat["lifetime_volume"]
    df_feat["raw_cell_cost_total"] = df_feat["battery_kwh"] * df_feat["cell_cost_per_kwh"]
    df_feat["cell_cost_to_msrp_ratio"] = df_feat["raw_cell_cost_total"] / df_feat["base_msrp"]
    df_feat["eco_density_per_supplier"] = df_feat["num_engineering_change_orders"] / df_feat["num_tier1_suppliers"]
    df_feat["regulatory_compliance_composite"] = (
        (df_feat["mineral_fta_sourcing_pct"] * 0.5) + (df_feat["battery_na_assembly_pct"] * 0.5)
    ) * (1 - df_feat["feoc_breach_flag"])
    
    # Categorical One-Hot Encoding
    categorical_cols = ["platform", "trim", "assembly_plant"]
    df_feat = pd.get_dummies(df_feat, columns=categorical_cols, drop_first=True)
    
    return df_feat
