"""
portfolio_dcf_engine.py - Automated Deliverables O1 & O2: Program DCF & 8-Level Variance Bridge
Author: Bhupesh Chandra Dimri
Target: GM Corporate Strategy & GBCT Gateway Reviews
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List

class PortfolioDCFEngine:
    """
    Automated Deliverable O1 & O2 Generator:
    - O1: Multi-year vehicle platform discounted cash flow (DCF), NPV, IRR, and tooling amortization schedules.
    - O2: Executive 8-Level Strategy Variance Bridge (Price, Volume, Mix, Raw Material, Tooling D&A, 45X, 30D).
    """
    def __init__(self, discount_rate: float = 0.10):
        self.discount_rate = discount_rate

    def generate_program_gateway_dcf(
        self,
        program_name: str,
        annual_volume: int,
        lifecycle_years: int,
        msrp: float,
        unit_bom: float,
        tooling_capex: float,
        battery_kwh: float,
        sec_45x_offset: float,
        variable_overhead_per_unit: float = 3800.0,
        direct_labor_per_unit: float = 2400.0
    ) -> Dict[str, Any]:
        """
        Deliverable O1: Program Gateway Review Financial Profile.
        """
        lifetime_units = annual_volume * lifecycle_years
        
        # Units-of-production tooling amortization
        tooling_amort_per_unit = tooling_capex / lifetime_units
        
        # Net unit COGS (BOM + Labor + Overhead + Tooling D&A - Section 45X credit)
        gross_unit_cogs = unit_bom + direct_labor_per_unit + variable_overhead_per_unit + tooling_amort_per_unit
        net_unit_cogs = gross_unit_cogs - sec_45x_offset
        unit_contribution_margin = msrp - net_unit_cogs
        margin_pct = (unit_contribution_margin / msrp) * 100.0
        
        # Cash Flow Profile
        # Year 0: Tooling CapEx outlay
        # Years 1-N: Annual Operating Cash Flows
        annual_ebitda = annual_volume * unit_contribution_margin
        cash_flows = [-tooling_capex] + [annual_ebitda] * lifecycle_years
        
        # Financial Metrics
        npv = float(np.npv(self.discount_rate, cash_flows)) if hasattr(np, 'npv') else self._calculate_npv(self.discount_rate, cash_flows)
        irr = float(self._calculate_irr(cash_flows))
        payback_years = float(tooling_capex / annual_ebitda) if annual_ebitda > 0 else 99.0
        
        return {
            "program_name": program_name,
            "annual_volume": annual_volume,
            "lifetime_units": lifetime_units,
            "msrp": msrp,
            "unit_bom": unit_bom,
            "tooling_capex": tooling_capex,
            "tooling_amort_per_unit": tooling_amort_per_unit,
            "sec_45x_offset": sec_45x_offset,
            "net_unit_cogs": net_unit_cogs,
            "unit_contribution_margin": unit_contribution_margin,
            "margin_pct": margin_pct,
            "annual_ebitda": annual_ebitda,
            "lifetime_ebitda": annual_ebitda * lifecycle_years,
            "npv_10pct": npv,
            "irr_pct": irr * 100.0,
            "discounted_payback_years": payback_years,
            "gateway_recommendation": "Green Gate (Approved)" if npv > 0 and irr >= 0.12 else "Amber Gate (Conditional)"
        }

    def generate_8level_variance_bridge(
        self,
        baseline_units: int,
        actual_units: int,
        baseline_msrp: float,
        actual_msrp: float,
        baseline_bom: float,
        actual_bom: float,
        baseline_tooling: float,
        actual_tooling: float,
        baseline_45x: float,
        actual_45x: float
    ) -> Dict[str, Any]:
        """
        Deliverable O2: Corporate Strategy 8-Level Variance Bridge.
        Deconstructs total platform EBITDA variance into distinct management levers.
        """
        base_cm = baseline_msrp - baseline_bom - (baseline_tooling / baseline_units) + baseline_45x
        act_cm = actual_msrp - actual_bom - (actual_tooling / actual_units) + actual_45x
        
        base_ebitda = baseline_units * base_cm
        act_ebitda = actual_units * act_cm
        total_variance = act_ebitda - base_ebitda
        
        # 8-Level Variance Decomposition
        volume_variance = (actual_units - baseline_units) * base_cm
        price_variance = actual_units * (actual_msrp - baseline_msrp)
        raw_material_bom_variance = -1.0 * actual_units * (actual_bom - baseline_bom)
        tooling_da_variance = -1.0 * actual_units * ((actual_tooling / actual_units) - (baseline_tooling / baseline_units))
        sec_45x_tax_variance = actual_units * (actual_45x - baseline_45x)
        mix_and_other_variance = total_variance - (
            volume_variance + price_variance + raw_material_bom_variance + tooling_da_variance + sec_45x_tax_variance
        )
        
        waterfall_steps = [
            {"label": "Baseline Program EBITDA", "value": base_ebitda, "type": "baseline"},
            {"label": "Volume Impact", "value": volume_variance, "type": "variance"},
            {"label": "Net Price Realization", "value": price_variance, "type": "variance"},
            {"label": "BOM & Raw Material Inflation", "value": raw_material_bom_variance, "type": "variance"},
            {"label": "Tooling D&A Overrun", "value": tooling_da_variance, "type": "variance"},
            {"label": "IRA Section 45X Credit Delta", "value": sec_45x_tax_variance, "type": "variance"},
            {"label": "Mix / Operational Levers", "value": mix_and_other_variance, "type": "variance"},
            {"label": "Final Realized EBITDA", "value": act_ebitda, "type": "total"}
        ]
        
        return {
            "baseline_ebitda": base_ebitda,
            "actual_ebitda": act_ebitda,
            "total_variance": total_variance,
            "waterfall_steps": waterfall_steps
        }

    @staticmethod
    def _calculate_npv(rate: float, values: List[float]) -> float:
        return sum(v / ((1.0 + rate) ** t) for t, v in enumerate(values))

    @staticmethod
    def _calculate_irr(values: List[float], max_iter: int = 1000) -> float:
        rate = 0.1
        for _ in range(max_iter):
            npv = sum(v / ((1.0 + rate) ** t) for t, v in enumerate(values))
            d_npv = sum(-t * v / ((1.0 + rate) ** (t + 1)) for t, v in enumerate(values))
            if abs(d_npv) < 1e-9:
                break
            new_rate = rate - npv / d_npv
            if abs(new_rate - rate) < 1e-6:
                return new_rate
            rate = new_rate
        return rate
