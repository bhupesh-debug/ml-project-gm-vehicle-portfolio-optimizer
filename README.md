# ML Project: Autonomous Vehicle Program Business Case & Regulatory Portfolio Optimizer (`ml-project-gm-vehicle-portfolio-optimizer`)

> **Enterprise Machine Learning & Financial Engineering Engine for Automotive Capital Allocation, Vehicle Line BOM Cost & Tooling Overrun Prediction, and Inflation Reduction Act (IRA Section 30D / 45X) Portfolio Optimization.**  
> Built for automotive corporate strategy, vehicle line program finance, and platform gate governance (General Motors).  
> **Author:** Bhupesh Chandra Dimri | M.S. Accounting Analytics (Pace Lubin) | MBA Finance

---

## 🚀 Executive Summary & Strategic Value Proposition

In the automotive industry, multi-billion-dollar vehicle platform decisions face severe margin uncertainty due to:
1. **Tooling & BOM Cost Creep:** Initial engineering estimates in `eGPPS` diverge from final contracts due to commodity volatility (lithium, battery cells, steel) and unbilled engineering change orders (ECOs).
2. **Regulatory & Policy Volatility:** Determining consumer tax credit eligibility under **IRS Section 30D** (requiring 60%+ critical mineral and 70%+ battery component sourcing with 0% Foreign Entity of Concern content) and quantifying **Section 45X** advanced manufacturing production tax credits ($45/kWh: $35 cell + $10 module) directly dictate platform profitability.
3. **Units-of-Production Tooling Amortization:** Miscalculating multi-year volume baselines from `GLPT` distorts per-unit contribution margins during gateway milestone reviews.

`ml-project-gm-vehicle-portfolio-optimizer` automates and stabilizes this financial architecture:
* **ML Engine M1 (BOM & Tooling Regressors):** Gradient Boosting predicting realized vehicle BOM piece cost with **2.09% MAPE** ($R^2 = 0.9887$) and dedicated tooling CapEx expenditure with **2.54% MAPE** ($R^2 = 0.9862$).
* **ML Engine M2 (Regulatory Sourcing Classifier):** Random Forest classifier scoring Section 30D compliance with **99.17% accuracy** and automating Section 45X COGS offsets ($45/kWh) to dynamically output net contribution margins.
* **Deliverable O1 (Program Gateway Review DCF):** Automated 7-year multi-year platform cash flows, units-of-production tooling D&A, platform NPV @ 10%, IRR, and discounted payback schedules for the Global Business Case Tool (`GBCT`).
* **Deliverable O2 (Corporate Strategy 8-Level Variance Bridge):** Automated executive waterfall decomposing EBITDA variances into Volume, Price, BOM Inflation, Tooling Overruns, Section 45X Credits, and Mix levers.

---

## 🛠️ Architecture & System Data Flow

```
┌────────────────────────────────────────────────────────┐
│             MULTI-STREAM DATA INGESTION                │
├──────────────────────────┬─────────────────────────────┤
│  [D1] GLPT Volumes       │ 7-year lifecycle curves     │
│  [D2] eGPPS Contract BOM │ Tier-1 piece quotes & SOWs  │
│  [D3] Policy Feeds       │ IRA 30D/45X & FEOC lists    │
└──────────────────────────┴─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│                AUTONOMOUS ML ENGINES                   │
├──────────────────────────┬─────────────────────────────┤
│  [M1] Cost & Tooling Reg │ Gradient Boosting (2.1% err)│
│  [M2] Regulatory Engine  │ Random Forest (99.2% Acc)   │
└──────────────────────────┴─────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────┐
│           AUTOMATED ACCOUNTING DELIVERABLES            │
├──────────────────────────┬─────────────────────────────┤
│  [O1] Program Gateway DCF│ Units-of-Production D&A,    │
│                          │ Net Margin %, NPV @ 10%     │
│  [O2] 8-Level Strategy   │ Price • Volume • Mix • Raw  │
│       Variance Bridge    │ Mat • Tooling • Tax Credits │
└──────────────────────────┴─────────────────────────────┘
```

---

## 💻 Tech Stack & Dependencies

* **Core Language:** Python 3.10+
* **Machine Learning:** Scikit-Learn, XGBoost, NumPy, Pandas
* **Financial Modeling:** Units-of-Production Amortization, Discounted Cash Flows (DCF), Net Present Value (NPV), Internal Rate of Return (IRR), Monte Carlo Simulation
* **Interactive Frontend:** HTML5, CSS3 Glassmorphism, Vanilla JavaScript, Responsive Design, Live Client-Side Waterfall & DCF Recalculation

---

## ⚡ Quickstart & Execution

```bash
# 1. Clone the repository
git clone https://github.com/bhupesh-debug/ml-project-gm-vehicle-portfolio-optimizer.git
cd ml-project-gm-vehicle-portfolio-optimizer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the complete end-to-end ML & financial engineering demonstration
python generate_demo.py
```

### Live Interactive Web Hub
Open `index.html` in any web browser to access the real-time simulation dashboard with:
* Interactive architectural flowchart mapping D1, D2, D3 $\to$ M1, M2 $\to$ O1, O2
* Real-time sliders for vehicle MSRP, GLPT volumes, cell $/kWh, tooling CapEx, and FTA sourcing %
* Live-updating executive 8-level variance waterfall bridge and 7-year DCF schedule
* Monte Carlo portfolio simulation engine (1,000 runs)

---

## 📊 Sample Model Performance & Gateway Results

| Metric | Machine Learning Model | Realized Score | Industry Benchmark |
| :--- | :--- | :--- | :--- |
| **BOM Piece Cost Prediction** | Gradient Boosting Regressor | **2.09% MAPE** | $< 5.0\%$ target |
| **BOM Model Goodness of Fit** | Gradient Boosting Regressor | **$R^2 = 0.9887$** | $> 0.95$ target |
| **Tooling CapEx Overrun** | Random Forest Regressor | **2.54% MAPE** | $< 5.0\%$ target |
| **Section 30D Tax Credit Tier** | Random Forest Classifier | **99.17% Accuracy** | $> 95.0\%$ target |

### Program Gateway Review DCF (Project Titan Example)
* **Platform:** Next-Gen Ultium Midsize CUV (7-Year Lifecycle: 595,000 units)
* **Dedicated Tooling Amortization (UOP):** $\$873.95/\text{unit}$ ($\$520\text{M} / 595\text{k}$)
* **Section 45X Offset:** $-\$3,825/\text{unit}$ ($85\text{ kWh} \times \$45/\text{kWh}$)
* **Net Realized COGS:** $\$35,748.95/\text{unit}$
* **Unit Contribution Margin:** $\$18,251.05/\text{unit}$ (**33.80% Gross Margin**)
* **Platform NPV @ 10%:** **$\$7,032.57\text{M}$** | **IRR: 298.32%** | **Payback: 0.34 years**
* **Executive Recommendation:** **Green Gate (Approved)**

---

## 👤 Author Profile
**Bhupesh Chandra Dimri**  
*M.S. in Accounting Analytics — Pace University (Lubin School of Business)*  
*MBA in Finance*  
*Expertise:* Enterprise Automotive Financial Modeling, General Ledger Close, Capital Allocation, Regulatory Compliance (IRA 30D/45X), Machine Learning Automation.
