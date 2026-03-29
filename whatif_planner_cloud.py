"""
=============================================================================
PepsiCo Beverage Analytics — Price & Promotion What-If Planner
Streamlit Community Cloud Edition  |  Self-contained (no local files needed)
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PepsiCo Price & Promo Planner",
    page_icon="🥤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# BRANDING & CSS
# ─────────────────────────────────────────────────────────────────────────────
NAVY  = "#003087"
RED   = "#E31837"
GREEN = "#1a7a45"
AMBER = "#d97706"
LGREY = "#f4f6f9"
WHITE = "#ffffff"

st.markdown(f"""
<style>
  /* Global font */
  html, body, [class*="css"] {{ font-family: 'Segoe UI', Tahoma, sans-serif; }}

  /* Background */
  [data-testid="stAppViewContainer"] {{ background: {LGREY}; }}
  [data-testid="stSidebar"] {{ background: {NAVY} !important; }}
  [data-testid="stSidebar"] * {{ color: {WHITE} !important; }}
  [data-testid="stSidebar"] .stButton > button {{
      background: {RED}; color: white; border: none;
      border-radius: 6px; font-weight: 700; width: 100%; margin-bottom: 6px;
  }}
  [data-testid="stSidebar"] .stButton > button:hover {{ background: #b01429; }}

  /* Metric cards */
  .kpi-card {{
      background: {WHITE}; border-radius: 10px;
      padding: 18px 16px; text-align: center;
      box-shadow: 0 2px 8px rgba(0,0,0,0.07);
  }}
  .kpi-label {{ font-size: 11px; font-weight: 700; text-transform: uppercase;
               letter-spacing: 1px; color: #888; margin-bottom: 6px; }}
  .kpi-value {{ font-size: 28px; font-weight: 800; margin-bottom: 2px; }}
  .kpi-sub   {{ font-size: 12px; color: #999; }}

  /* Scenario badge */
  .scenario-badge {{
      display: inline-block; padding: 6px 14px; border-radius: 20px;
      font-weight: 700; font-size: 13px; margin-bottom: 10px;
  }}

  /* Table */
  .styled-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .styled-table th {{ background: {NAVY}; color: white; padding: 8px 12px; text-align: left; }}
  .styled-table td {{ padding: 8px 12px; border-bottom: 1px solid #eee; }}
  .styled-table tr:nth-child(even) td {{ background: #f7f9fc; }}

  /* Insight box */
  .insight-box {{
      background: {WHITE}; border-left: 4px solid {NAVY};
      padding: 14px 16px; border-radius: 6px;
      margin: 8px 0; font-size: 13px; line-height: 1.6;
  }}
  .warning-box {{
      background: #fff8e1; border-left: 4px solid {AMBER};
      padding: 12px 16px; border-radius: 6px;
      margin: 8px 0; font-size: 13px;
  }}
  .success-box {{
      background: #e8f5e9; border-left: 4px solid {GREEN};
      padding: 12px 16px; border-radius: 6px;
      margin: 8px 0; font-size: 13px;
  }}

  /* Hide streamlit menu in prod */
  #MainMenu, footer {{ visibility: hidden; }}

  /* Tabs */
  .stTabs [data-baseweb="tab"] {{ font-weight: 600; font-size: 14px; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDED MODEL DATA  (from OLS run on ADUSA Corp-RMA – Groc, 2023-2024)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_DATA = [
    {
        "label":              "Pepsi 12oz×12",
        "product":            "CSD CAN 12 OZ 12 CT PEPSI",
        "category":           "Cola",
        "brand":              "PepsiCo",
        "avg_unit_sales":     337_780,
        "avg_price":          6.03,
        "avg_tpr_depth":      13.9,          # Updated: Window=13 rolling max (was 28.4 W=7 median)
        "promo_freq_pct":     80.0,
        "edlp_elasticity":   -1.7453,        # Updated: Window=13 (was -2.3976)
        "promo_elasticity":   0.01968,        # Updated: 1.9680÷100 for 0-100 input compat (was 0.02977)
        "cross_price_elast": -0.528,          # Updated: vs Coke, not sig (p=0.31) (was +1.345)
        "competitor":         "Coke 12oz×12",
        "r2":                 0.8671,          # Updated: Window=13 model (was 0.9564)
        "cv_r2":              0.3774,          # Updated: 5-fold CV (was 0.9434)
        "strategy":           "EDLP",
        "strategy_color":     NAVY,
    },
    {
        "label":              "Pepsi 12oz×24",
        "product":            "CSD CAN 12 OZ 24 CT PEPSI",
        "category":           "Cola",
        "brand":              "PepsiCo",
        "avg_unit_sales":     28_742,
        "avg_price":          11.95,
        "avg_tpr_depth":      13.4,
        "promo_freq_pct":     38.1,
        "edlp_elasticity":   -3.0655,
        "promo_elasticity":   0.04212,
        "cross_price_elast":  0.100,
        "competitor":         "Coke 12oz×24",
        "r2":                 0.9610,
        "cv_r2":              0.9033,
        "strategy":           "Hybrid",
        "strategy_color":     AMBER,
    },
    {
        "label":              "Pepsi 7.5oz×6",
        "product":            "CSD CAN 7.5 OZ 6 CT PEPSI",
        "category":           "Cola",
        "brand":              "PepsiCo",
        "avg_unit_sales":     22_524,
        "avg_price":          4.56,
        "avg_tpr_depth":      7.5,
        "promo_freq_pct":     17.1,
        "edlp_elasticity":   -0.7495,
        "promo_elasticity":   0.00833,
        "cross_price_elast": -0.476,
        "competitor":         "Coke 12oz×12",
        "r2":                 0.9583,
        "cv_r2":              0.9247,
        "strategy":           "Distribution-First",
        "strategy_color":     GREEN,
    },
    {
        "label":              "Coke 12oz×12",
        "product":            "CSD CAN 12 OZ 12 CT COKE",
        "category":           "Cola",
        "brand":              "Coca-Cola",
        "avg_unit_sales":     460_289,
        "avg_price":          6.70,
        "avg_tpr_depth":      21.5,
        "promo_freq_pct":     79.0,
        "edlp_elasticity":   -2.3770,
        "promo_elasticity":   0.03173,
        "cross_price_elast": -0.164,
        "competitor":         "Pepsi 12oz×12",
        "r2":                 0.9719,
        "cv_r2":              0.9609,
        "strategy":           "EDLP",
        "strategy_color":     NAVY,
    },
    {
        "label":              "Coke 12oz×24",
        "product":            "CSD CAN 12 OZ 24 CT COKE",
        "category":           "Cola",
        "brand":              "Coca-Cola",
        "avg_unit_sales":     43_422,
        "avg_price":          12.98,
        "avg_tpr_depth":      7.1,
        "promo_freq_pct":     17.1,
        "edlp_elasticity":   -1.6628,
        "promo_elasticity":   0.02893,
        "cross_price_elast": -0.682,
        "competitor":         "Pepsi 12oz×24",
        "r2":                 0.9215,
        "cv_r2":              0.8874,
        "strategy":           "Hi-Lo",
        "strategy_color":     RED,
    },
    {
        "label":              "Mountain Dew 12oz×12",
        "product":            "CSD CAN 12 OZ 12 CT MOUNTAIN DEW",
        "category":           "Citrus",
        "brand":              "PepsiCo",
        "avg_unit_sales":     222_271,
        "avg_price":          6.08,
        "avg_tpr_depth":      12.2,          # Updated: Window=13 rolling max (was 31.7 W=7 median)
        "promo_freq_pct":     72.4,
        "edlp_elasticity":   -1.6365,        # Updated: Window=13 (was -2.2662)
        "promo_elasticity":   0.01713,        # Updated: 1.7127÷100 for 0-100 input compat (was 0.02636)
        "cross_price_elast":  0.000,          # Updated: vs Sprite, LASSO dropped (was +0.965)
        "competitor":         "Sprite 12oz×12",
        "r2":                 0.8715,          # Updated: Window=13 model (was 0.9697)
        "cv_r2":              0.4513,          # Updated: 5-fold CV (was 0.9624)
        "strategy":           "Promo-Led",
        "strategy_color":     AMBER,
    },
    {
        "label":              "Sprite 12oz×12",
        "product":            "CSD CAN 12 OZ 12 CT SPRITE",
        "category":           "Citrus",
        "brand":              "Coca-Cola",
        "avg_unit_sales":     124_731,
        "avg_price":          6.67,
        "avg_tpr_depth":      22.6,
        "promo_freq_pct":     75.2,
        "edlp_elasticity":   -2.3607,
        "promo_elasticity":   0.03097,
        "cross_price_elast": -0.302,
        "competitor":         "Mountain Dew 12oz×12",
        "r2":                 0.9600,
        "cv_r2":              0.9450,
        "strategy":           "EDLP",
        "strategy_color":     NAVY,
    },
    {
        "label":              "Sprite 7.5oz×6",
        "product":            "CSD CAN 7.5 OZ 6 CT SPRITE",
        "category":           "Citrus",
        "brand":              "Coca-Cola",
        "avg_unit_sales":     17_843,
        "avg_price":          4.64,
        "avg_tpr_depth":      7.2,
        "promo_freq_pct":     13.3,
        "edlp_elasticity":   -1.6385,
        "promo_elasticity":   0.01242,
        "cross_price_elast": -0.597,
        "competitor":         "Mountain Dew 12oz×12",
        "r2":                 0.7158,
        "cv_r2":              0.5839,
        "strategy":           "Distribution-First",
        "strategy_color":     GREEN,
    },
]

LABELS = [d["label"] for d in MODEL_DATA]
PEPSI_SKUS = [d["label"] for d in MODEL_DATA if d["brand"] == "PepsiCo"]


# ─────────────────────────────────────────────────────────────────────────────
# PRE-BUILT DEMO SCENARIOS
# ─────────────────────────────────────────────────────────────────────────────
DEMO_SCENARIOS = {
    "🌞 Summer Promo Blitz": {
        "description": (
            "It's Q3 peak season. Run a coordinated 30% TPR on Pepsi 12oz×12 "
            "and a 28% TPR on Mountain Dew to capture maximum summer volume "
            "while Sprite and Coke hold their prices steady."
        ),
        "strategy_highlight": "Promo-Led (Hi-Lo) in peak season",
        "edlp_changes": {
            "Pepsi 12oz×12": 0.0,
            "Mountain Dew 12oz×12": 0.0,
        },
        "tpr_changes": {
            "Pepsi 12oz×12": 30.0,
            "Mountain Dew 12oz×12": 28.0,
            "Pepsi 12oz×24": 0.0,
            "Pepsi 7.5oz×6": 0.0,
        },
        "color": "#E31837",
    },
    "💰 EDLP Price Cut — Pepsi vs Coke": {
        "description": (
            "Reduce Pepsi 12oz×12 shelf price by 3% below Coke's current level. "
            "This exploits the strong Pepsi–Coke substitute relationship: "
            "lower Pepsi shelf price draws Coke buyers, gaining incremental share "
            "without relying on temporary promotions."
        ),
        "strategy_highlight": "EDLP — Permanent shelf-price advantage",
        "edlp_changes": {
            "Pepsi 12oz×12": -3.0,
            "Mountain Dew 12oz×12": 0.0,
        },
        "tpr_changes": {
            "Pepsi 12oz×12": 0.0,
            "Mountain Dew 12oz×12": 0.0,
            "Pepsi 12oz×24": 0.0,
            "Pepsi 7.5oz×6": 0.0,
        },
        "color": "#003087",
    },
    "🚀 Portfolio Optimisation": {
        "description": (
            "Multi-lever portfolio play: (1) Hold Pepsi 12oz×12 EDLP steady with a "
            "25% targeted TPR event. (2) Test a +5% price increase on Pepsi 7.5oz×6 "
            "(mini-cans are Giffen-adjacent — buyers tolerate premium pricing). "
            "(3) Run a 25% Mtn Dew TPR to time with Sprite's higher regular price."
        ),
        "strategy_highlight": "Portfolio — Mix of EDLP + Selective Hi-Lo + Premium",
        "edlp_changes": {
            "Pepsi 12oz×12": 0.0,
            "Pepsi 7.5oz×6": +5.0,
            "Mountain Dew 12oz×12": 0.0,
        },
        "tpr_changes": {
            "Pepsi 12oz×12": 25.0,
            "Mountain Dew 12oz×12": 25.0,
            "Pepsi 12oz×24": 0.0,
            "Pepsi 7.5oz×6": 0.0,
        },
        "color": "#1a7a45",
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# CALCULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def calc_impact(sku_data: dict, price_change_pct: float, tpr_depth: float,
                trade_cost_per_unit: float = 0.10) -> dict:
    """
    Compute volume, revenue, and ROI impact for a single SKU.

    Price change: log-log elasticity → vol_change% = edlp_elast × price_pct_change
    Promo change: semi-log → vol_change% = promo_elast × tpr_depth_pct
    """
    base_vol = sku_data["avg_unit_sales"]
    base_price = sku_data["avg_price"]
    base_rev  = base_vol * base_price

    # EDLP volume change
    edlp_vol_chg_pct = sku_data["edlp_elasticity"] * price_change_pct / 100.0

    # TPR volume change
    tpr_vol_chg_pct = sku_data["promo_elasticity"] * tpr_depth

    # Combined volume change (additive in % terms for small changes)
    total_vol_chg_pct = edlp_vol_chg_pct + tpr_vol_chg_pct
    new_vol = base_vol * (1.0 + total_vol_chg_pct)
    vol_chg = new_vol - base_vol

    # Effective price (shelf price × (1 − tpr_depth/100) if on promo)
    effective_price = base_price * (1.0 + price_change_pct / 100.0)
    if tpr_depth > 0:
        effective_price = effective_price * (1.0 - tpr_depth / 100.0)

    new_rev = new_vol * effective_price
    rev_chg = new_rev - base_rev

    # Trade investment (trade cost × incremental volume)
    incr_vol = max(vol_chg, 0)
    trade_invest = incr_vol * trade_cost_per_unit
    net_rev = rev_chg - trade_invest
    roi_pct = (net_rev / trade_invest * 100) if trade_invest > 0 else 0.0

    return {
        "base_vol":        base_vol,
        "new_vol":         new_vol,
        "vol_chg":         vol_chg,
        "vol_chg_pct":     total_vol_chg_pct * 100,
        "edlp_vol_pct":    edlp_vol_chg_pct * 100,
        "tpr_vol_pct":     tpr_vol_chg_pct * 100,
        "base_rev":        base_rev,
        "new_rev":         new_rev,
        "rev_chg":         rev_chg,
        "trade_invest":    trade_invest,
        "net_rev":         net_rev,
        "roi_pct":         roi_pct,
        "base_price":      base_price,
        "effective_price": effective_price,
    }


def fmt_units(v: float) -> str:
    if abs(v) >= 1_000_000: return f"{v/1_000_000:.2f}M"
    if abs(v) >= 1_000:     return f"{v/1_000:.1f}K"
    return f"{v:.0f}"

def fmt_usd(v: float) -> str:
    if abs(v) >= 1_000_000: return f"${v/1_000_000:.2f}M"
    if abs(v) >= 1_000:     return f"${v/1_000:.1f}K"
    return f"${v:.0f}"

def fmt_pct(v: float) -> str:
    return f"{v:+.1f}%"

def color_val(v: float, reverse: bool = False) -> str:
    if reverse:
        return GREEN if v < 0 else (RED if v > 0 else "#555")
    return GREEN if v > 0 else (RED if v < 0 else "#555")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; padding: 10px 0 20px 0;">
      <div style="font-size:28px; font-weight:900; letter-spacing:2px;">PepsiCo</div>
      <div style="font-size:11px; opacity:0.7; letter-spacing:1px; text-transform:uppercase;">
        Price & Promotion Planner
      </div>
      <hr style="border-color: rgba(255,255,255,0.2); margin: 12px 0;">
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Load Demo Scenario")
    st.markdown(
        "<div style='font-size:12px; opacity:0.8; margin-bottom:12px;'>"
        "Click any scenario to instantly populate all levers with real recommendations.</div>",
        unsafe_allow_html=True)

    for name in DEMO_SCENARIOS:
        if st.button(name, key=f"btn_{name}"):
            st.session_state["active_scenario"] = name

    st.markdown("<hr style='border-color: rgba(255,255,255,0.2); margin: 16px 0;'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Settings")
    trade_cost = st.slider("Trade Cost per Unit ($)", 0.0, 0.50, 0.10, 0.01,
                            help="Estimated cost of promotional trade investment per incremental unit sold.")
    st.markdown("<hr style='border-color: rgba(255,255,255,0.2); margin: 16px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px; opacity:0.65; line-height:1.6;'>
    <b>Data:</b> ADUSA Corp-RMA – Groc<br>
    <b>Period:</b> Jan 2023 – Dec 2024<br>
    <b>Model:</b> Log-Log OLS (statsmodels)<br>
    <b>SKUs:</b> 8 CSD multipacks<br><br>
    Elasticities are week-level, store-level estimates.
    Results shown are indicative — validate before final pricing decisions.
    </div>
    """, unsafe_allow_html=True)

# Initialise session state
if "active_scenario" not in st.session_state:
    st.session_state["active_scenario"] = None

# ─────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="background: linear-gradient(135deg, {NAVY} 0%, #001a4d 100%);
            padding: 28px 32px; border-radius: 12px; margin-bottom: 24px;
            color: white;">
  <div style="font-size: 26px; font-weight: 900; letter-spacing: 1px;">
    🥤 PepsiCo Price & Promotion What-If Planner
  </div>
  <div style="font-size: 14px; opacity: 0.8; margin-top: 6px;">
    Instantly model the volume and revenue impact of any pricing or promotional change
    across the CSD portfolio · ADUSA Corp-RMA – Grocery · 2023–2024 · Powered by OLS Elasticity Models
  </div>
</div>
""", unsafe_allow_html=True)

# Show active scenario banner
active = st.session_state.get("active_scenario")
if active:
    sc = DEMO_SCENARIOS[active]
    st.markdown(f"""
    <div style="background: {sc['color']}18; border: 2px solid {sc['color']};
                border-radius: 10px; padding: 16px 20px; margin-bottom: 20px;">
      <span style="font-size:16px; font-weight:800; color:{sc['color']};">
        {active}
      </span>
      <br><span style="font-size:13px; color:#444; line-height:1.6; display:inline-block; margin-top:6px;">
        {sc['description']}
      </span>
      <br><span style="display:inline-block; margin-top:8px; background:{sc['color']};
                       color:white; padding:3px 12px; border-radius:12px; font-size:11px; font-weight:700;">
        {sc['strategy_highlight']}
      </span>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_edlp, tab_promo, tab_portfolio, tab_about = st.tabs([
    "📊 Shelf Price (EDLP)",
    "🏷️ Promotions (TPR)",
    "🗂️ Portfolio Summary",
    "ℹ️ How to Use",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — EDLP / SHELF PRICE
# ─────────────────────────────────────────────────────────────────────────────
with tab_edlp:
    st.markdown("#### What happens if we change the regular shelf price?")
    st.markdown(
        "<div style='font-size:13px; color:#555; margin-bottom:20px;'>"
        "Use the slider to raise or lower the <b>everyday shelf price</b> (not a promotion). "
        "The model shows the expected volume and revenue impact based on what shoppers actually did in 2023-2024.</div>",
        unsafe_allow_html=True)

    # Get scenario defaults
    sc_edlp = {}
    if active:
        sc_edlp = DEMO_SCENARIOS[active].get("edlp_changes", {})

    # SKU selector
    selected_sku = st.selectbox("Select Product", LABELS, key="edlp_sku")
    sku_data = next(d for d in MODEL_DATA if d["label"] == selected_sku)

    # Determine default value from scenario
    default_edlp = float(sc_edlp.get(selected_sku, 0.0))

    col_slider, col_info = st.columns([2, 1])
    with col_slider:
        price_chg = st.slider(
            "📐 Shelf Price Change (%)",
            min_value=-20.0, max_value=20.0, value=default_edlp, step=0.5,
            help="Positive = price increase. Negative = price reduction.",
            key=f"edlp_slider_{selected_sku}",
        )

    with col_info:
        base_p = sku_data["avg_price"]
        new_p  = base_p * (1 + price_chg / 100)
        st.markdown(f"""
        <div class="insight-box" style="margin-top:10px;">
          <b>Price Change Details</b><br>
          Current avg price: <b>${base_p:.2f}</b><br>
          New shelf price: <b>${new_p:.2f}</b><br>
          EDLP Elasticity: <b>{sku_data['edlp_elasticity']:+.3f}</b><br>
          Recommended strategy: <b>{sku_data['strategy']}</b>
        </div>
        """, unsafe_allow_html=True)

    result = calc_impact(sku_data, price_chg, 0.0, trade_cost)

    # KPI Cards
    st.markdown("<br>", unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        vc = color_val(result["vol_chg"])
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Volume Change</div>
          <div class="kpi-value" style="color:{vc};">{fmt_pct(result['vol_chg_pct'])}</div>
          <div class="kpi-sub">{fmt_units(result['vol_chg'])} units/week</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        rc = color_val(result["rev_chg"])
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Revenue Impact</div>
          <div class="kpi-value" style="color:{rc};">{fmt_usd(result['rev_chg'])}</div>
          <div class="kpi-sub">per week</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">New Weekly Volume</div>
          <div class="kpi-value" style="color:{NAVY};">{fmt_units(result['new_vol'])}</div>
          <div class="kpi-sub">vs {fmt_units(result['base_vol'])} baseline</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        cross = sku_data["cross_price_elast"]
        competitor = sku_data["competitor"]
        comp_vol_pct = cross * price_chg / 100 * 100
        cc = color_val(-comp_vol_pct)  # positive cross + price increase → bad for us
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Competitor Impact</div>
          <div class="kpi-value" style="color:{cc};">{fmt_pct(comp_vol_pct)}</div>
          <div class="kpi-sub">{competitor} vol change</div>
        </div>""", unsafe_allow_html=True)

    # Chart — demand curve
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**📈 Demand Curve — Volume Response to Price**")
        price_range = np.linspace(-15, 15, 100)
        vol_pct_range = [sku_data["edlp_elasticity"] * p / 100 * 100 for p in price_range]

        fig, ax = plt.subplots(figsize=(7, 4))
        fig.patch.set_facecolor("#fafbfd")
        ax.set_facecolor("#fafbfd")
        ax.plot(price_range, vol_pct_range, color=NAVY, linewidth=2.5)
        ax.axhline(0, color="#ccc", linewidth=0.8)
        ax.axvline(0, color="#ccc", linewidth=0.8)
        ax.scatter([price_chg], [sku_data["edlp_elasticity"] * price_chg / 100 * 100],
                   color=RED, zorder=5, s=80)
        ax.axvline(price_chg, color=RED, linestyle="--", linewidth=1.2, alpha=0.6)
        ax.fill_between(price_range, vol_pct_range,
                        where=[p < 0 for p in price_range],
                        alpha=0.08, color=GREEN)
        ax.fill_between(price_range, vol_pct_range,
                        where=[p > 0 for p in price_range],
                        alpha=0.08, color=RED)
        ax.set_xlabel("Shelf Price Change (%)", fontsize=10, color="#555")
        ax.set_ylabel("Volume Change (%)", fontsize=10, color="#555")
        ax.set_title(f"{selected_sku} — Price Elasticity: {sku_data['edlp_elasticity']:+.3f}",
                     fontsize=11, fontweight="bold", color=NAVY)
        ax.grid(axis="both", linestyle="--", alpha=0.3)
        ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with c2:
        # Elasticity comparison across all SKUs
        st.markdown("**Elasticity Comparison — All SKUs**")
        fig2, ax2 = plt.subplots(figsize=(4.5, 4))
        fig2.patch.set_facecolor("#fafbfd")
        ax2.set_facecolor("#fafbfd")
        lbls = [d["label"].replace(" 12oz×12", "").replace(" 12oz×24","×24").replace(" 7.5oz×6","×6") for d in MODEL_DATA]
        vals = [d["edlp_elasticity"] for d in MODEL_DATA]
        colors_bar = [RED if d["label"] == selected_sku else (NAVY if d["brand"] == "PepsiCo" else "#aaa")
                      for d in MODEL_DATA]
        bars = ax2.barh(lbls, vals, color=colors_bar, edgecolor="white")
        ax2.axvline(0, color="#ccc", linewidth=0.8)
        ax2.axvline(-1, color=AMBER, linestyle="--", linewidth=1, alpha=0.7, label="Elastic threshold")
        ax2.set_xlabel("EDLP Elasticity", fontsize=9)
        ax2.tick_params(labelsize=8)
        ax2.spines[["top","right"]].set_visible(False)
        ax2.set_facecolor("#fafbfd")
        legend_patches = [mpatches.Patch(color=NAVY, label="PepsiCo"),
                          mpatches.Patch(color="#aaa", label="Competitor"),
                          mpatches.Patch(color=RED, label="Selected")]
        ax2.legend(handles=legend_patches, fontsize=8, framealpha=0.5)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

    # Insight
    if price_chg == 0:
        st.markdown("""<div class="insight-box">
        👆 <b>Move the slider above</b> to model the impact of a price change, or load a Demo Scenario from the left panel.
        </div>""", unsafe_allow_html=True)
    elif price_chg > 0 and sku_data["edlp_elasticity"] < -1.5:
        st.markdown(f"""<div class="warning-box">
        ⚠️ <b>High Elasticity Warning:</b> {selected_sku} has a high EDLP elasticity of
        {sku_data['edlp_elasticity']:.2f}. A {price_chg:.0f}% price increase is expected to cause
        a <b>{abs(result['vol_chg_pct']):.1f}% volume decline</b>. Recommendation: hold price at
        current level or consider an EDLP strategy.
        </div>""", unsafe_allow_html=True)
    elif price_chg < 0:
        st.markdown(f"""<div class="success-box">
        ✅ <b>Price Reduction Opportunity:</b> A {abs(price_chg):.0f}% shelf price reduction
        for {selected_sku} is expected to drive <b>{abs(result['vol_chg_pct']):.1f}%
        incremental volume</b>, adding approximately <b>{fmt_units(result['vol_chg'])}
        units/week</b> to the basket.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="insight-box">
        ℹ️ A {price_chg:.0f}% shelf price change for {selected_sku} is projected to
        change volume by <b>{result['vol_chg_pct']:+.1f}%</b> and weekly revenue by
        <b>{fmt_usd(result['rev_chg'])}</b>.
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — TPR / PROMOTION
# ─────────────────────────────────────────────────────────────────────────────
with tab_promo:
    st.markdown("#### What's the volume lift from a promotional event?")
    st.markdown(
        "<div style='font-size:13px; color:#555; margin-bottom:20px;'>"
        "Set the <b>discount depth</b> (how deep the promotion is) for a PepsiCo product. "
        "The model estimates gross volume lift, trade cost, and true net ROI.</div>",
        unsafe_allow_html=True)

    sc_tpr = {}
    if active:
        sc_tpr = DEMO_SCENARIOS[active].get("tpr_changes", {})

    pepsi_sku = st.selectbox("Select PepsiCo Product", PEPSI_SKUS, key="tpr_sku")
    sku_data2 = next(d for d in MODEL_DATA if d["label"] == pepsi_sku)
    default_tpr = float(sc_tpr.get(pepsi_sku, 0.0))

    col_sl, col_info2 = st.columns([2, 1])
    with col_sl:
        tpr_depth = st.slider(
            "🏷️ Promotional Discount Depth (%)",
            min_value=0.0, max_value=40.0, value=default_tpr, step=1.0,
            help="E.g. 25% means the shelf price is 25% below the regular price during the event.",
            key=f"tpr_slider_{pepsi_sku}",
        )
    with col_info2:
        st.markdown(f"""
        <div class="insight-box" style="margin-top:10px;">
          <b>Promo Details</b><br>
          Regular price: <b>${sku_data2['avg_price']:.2f}</b><br>
          Promo price: <b>${sku_data2['avg_price'] * (1 - tpr_depth/100):.2f}</b><br>
          TPR Elasticity: <b>{sku_data2['promo_elasticity']:+.5f}</b><br>
          Avg promo freq: <b>{sku_data2['promo_freq_pct']:.0f}%</b> of weeks
        </div>
        """, unsafe_allow_html=True)

    result2 = calc_impact(sku_data2, 0.0, tpr_depth, trade_cost)

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        vc2 = color_val(result2["vol_chg"])
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Gross Volume Lift</div>
          <div class="kpi-value" style="color:{vc2};">{fmt_pct(result2['tpr_vol_pct'])}</div>
          <div class="kpi-sub">{fmt_units(result2['vol_chg'])} units/week</div>
        </div>""", unsafe_allow_html=True)
    with k2:
        rc2 = color_val(result2["rev_chg"])
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Net Revenue Change</div>
          <div class="kpi-value" style="color:{rc2};">{fmt_usd(result2['rev_chg'])}</div>
          <div class="kpi-sub">after price reduction</div>
        </div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Trade Investment</div>
          <div class="kpi-value" style="color:{AMBER};">{fmt_usd(result2['trade_invest'])}</div>
          <div class="kpi-sub">at ${trade_cost:.2f}/unit</div>
        </div>""", unsafe_allow_html=True)
    with k4:
        roi_col = GREEN if result2["roi_pct"] > 100 else (AMBER if result2["roi_pct"] > 0 else RED)
        st.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">Promo ROI</div>
          <div class="kpi-value" style="color:{roi_col};">{result2['roi_pct']:+.0f}%</div>
          <div class="kpi-sub">net rev ÷ trade invest</div>
        </div>""", unsafe_allow_html=True)

    # Promo Lift Chart
    st.markdown("<br>", unsafe_allow_html=True)
    c1p, c2p = st.columns([3, 2])
    with c1p:
        st.markdown("**📊 Volume Lift Response Curve**")
        depths = np.linspace(0, 40, 100)
        lifts  = [sku_data2["promo_elasticity"] * d * 100 for d in depths]

        fig3, ax3 = plt.subplots(figsize=(7, 4))
        fig3.patch.set_facecolor("#fafbfd")
        ax3.set_facecolor("#fafbfd")
        ax3.plot(depths, lifts, color=RED, linewidth=2.5)
        if tpr_depth > 0:
            ax3.scatter([tpr_depth], [sku_data2["promo_elasticity"] * tpr_depth * 100],
                        color=NAVY, zorder=5, s=80)
            ax3.axvline(tpr_depth, color=NAVY, linestyle="--", linewidth=1.2, alpha=0.6)
        ax3.fill_between(depths, lifts, alpha=0.08, color=RED)
        # Mark average TPR depth
        avg_d = sku_data2["avg_tpr_depth"]
        ax3.axvline(avg_d, color=AMBER, linestyle=":", linewidth=1.5,
                    label=f"Avg TPR depth {avg_d:.0f}%")
        ax3.set_xlabel("Promotional Discount Depth (%)", fontsize=10, color="#555")
        ax3.set_ylabel("Expected Volume Lift (%)", fontsize=10, color="#555")
        ax3.set_title(f"{pepsi_sku} — Promo Elasticity: {sku_data2['promo_elasticity']:+.5f}",
                      fontsize=11, fontweight="bold", color=NAVY)
        ax3.legend(fontsize=9)
        ax3.grid(axis="both", linestyle="--", alpha=0.3)
        ax3.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig3, use_container_width=True)
        plt.close()

    with c2p:
        st.markdown("**Promo Elasticity — PepsiCo SKUs**")
        pep_data = [d for d in MODEL_DATA if d["brand"] == "PepsiCo"]
        fig4, ax4 = plt.subplots(figsize=(4.5, 4))
        fig4.patch.set_facecolor("#fafbfd")
        ax4.set_facecolor("#fafbfd")
        plbls  = [d["label"].replace(" 12oz×12","").replace(" 12oz×24","×24").replace(" 7.5oz×6","×6")
                  for d in pep_data]
        pvals  = [d["promo_elasticity"] * 100 for d in pep_data]
        pcols  = [RED if d["label"] == pepsi_sku else NAVY for d in pep_data]
        ax4.barh(plbls, pvals, color=pcols, edgecolor="white")
        ax4.set_xlabel("Promo Elasticity (×100)", fontsize=9)
        ax4.tick_params(labelsize=8)
        ax4.spines[["top","right"]].set_visible(False)
        ax4.set_title("Lift per 1pp Discount", fontsize=10, color=NAVY)
        plt.tight_layout()
        st.pyplot(fig4, use_container_width=True)
        plt.close()

    # Pantry load warning
    if tpr_depth > 25:
        st.markdown(f"""<div class="warning-box">
        ⚠️ <b>Pantry-Loading Risk:</b> At {tpr_depth:.0f}% depth, shoppers may stock up heavily.
        Expect a demand dip in the week <i>after</i> the promotion. Avoid running consecutive
        promo weeks — data shows a negative pantry-lag coefficient for {pepsi_sku}.
        </div>""", unsafe_allow_html=True)
    elif tpr_depth > 0:
        st.markdown(f"""<div class="success-box">
        ✅ A <b>{tpr_depth:.0f}% TPR</b> on {pepsi_sku} is expected to drive
        <b>{fmt_pct(result2['tpr_vol_pct'])}</b> volume lift — approximately
        <b>{fmt_units(result2['vol_chg'])} additional units/week</b>.
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — PORTFOLIO SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
with tab_portfolio:
    st.markdown("#### Portfolio-Level Impact Summary")
    st.markdown(
        "<div style='font-size:13px; color:#555; margin-bottom:20px;'>"
        "Adjust shelf prices and promo depths for all products at once to see total "
        "portfolio-level volume and revenue impact. Load a Demo Scenario for a pre-filled view.</div>",
        unsafe_allow_html=True)

    # Build input grid
    sc_edlp_all = DEMO_SCENARIOS[active]["edlp_changes"] if active else {}
    sc_tpr_all  = DEMO_SCENARIOS[active]["tpr_changes"]  if active else {}

    rows = []
    with st.form("portfolio_form"):
        header_cols = st.columns([2.5, 1.5, 1.5, 1.5, 1.5, 1.5])
        for ci, h in enumerate(["Product", "Brand", "Price Chg (%)", "TPR Depth (%)", "Strategy", "R² | CV R²"]):
            header_cols[ci].markdown(
                f"<div style='font-size:12px; font-weight:700; color:{NAVY}; padding-bottom:6px;'>{h}</div>",
                unsafe_allow_html=True)

        inputs = {}
        for d in MODEL_DATA:
            lbl = d["label"]
            c1, c2, c3, c4, c5, c6 = st.columns([2.5, 1.5, 1.5, 1.5, 1.5, 1.5])
            c1.markdown(
                f"<div style='font-size:13px; font-weight:600; padding-top:8px;'>{lbl}</div>",
                unsafe_allow_html=True)
            brand_col = NAVY if d["brand"] == "PepsiCo" else "#888"
            c2.markdown(
                f"<div style='font-size:12px; color:{brand_col}; padding-top:8px; font-weight:600;'>"
                f"{d['brand']}</div>",
                unsafe_allow_html=True)
            default_p = float(sc_edlp_all.get(lbl, 0.0))
            default_t = float(sc_tpr_all.get(lbl, 0.0))
            pchg = c3.number_input("", min_value=-20.0, max_value=20.0,
                                    value=default_p, step=0.5, label_visibility="collapsed",
                                    key=f"port_p_{lbl}")
            tprd = c4.number_input("", min_value=0.0, max_value=40.0,
                                    value=default_t, step=1.0, label_visibility="collapsed",
                                    key=f"port_t_{lbl}")
            strat_col = d["strategy_color"]
            c5.markdown(
                f"<div style='background:{strat_col}; color:white; padding:5px 8px; "
                f"border-radius:4px; font-size:11px; font-weight:700; margin-top:4px; "
                f"text-align:center;'>{d['strategy']}</div>",
                unsafe_allow_html=True)
            r2_col  = GREEN if d["r2"] > 0.85 else (AMBER if d["r2"] > 0.75 else RED)
            cvr2_col = GREEN if d["cv_r2"] > 0.50 else (AMBER if d["cv_r2"] > 0.35 else RED)
            c6.markdown(
                f"<div style='font-size:12px; font-weight:700; padding-top:4px;'>"
                f"<span style='color:{r2_col};'>R2: {d['r2']:.3f}</span>"
                f"<span style='color:#aaa; margin:0 3px;'>|</span>"
                f"<span style='color:{cvr2_col};'>CV: {d['cv_r2']:.3f}</span>"
                f"</div>",
                unsafe_allow_html=True)
            inputs[lbl] = {"price_chg": pchg, "tpr_depth": tprd}

        submitted = st.form_submit_button("🔄 Calculate Portfolio Impact",
                                          use_container_width=True)

    if submitted or active:
        results_all = []
        for d in MODEL_DATA:
            lbl   = d["label"]
            pchg  = inputs[lbl]["price_chg"]
            tprd  = inputs[lbl]["tpr_depth"]
            r     = calc_impact(d, pchg, tprd, trade_cost)
            results_all.append({
                "Product":          lbl,
                "Brand":            d["brand"],
                "Price Chg %":      pchg,
                "TPR Depth %":      tprd,
                "Vol Change %":     round(r["vol_chg_pct"], 1),
                "Vol Change (K)":   round(r["vol_chg"] / 1000, 1),
                "Base Vol (K)":     round(r["base_vol"] / 1000, 1),
                "New Vol (K)":      round(r["new_vol"] / 1000, 1),
                "Rev Change ($K)":  round(r["rev_chg"] / 1000, 1),
                "Trade Invest ($K)": round(r["trade_invest"] / 1000, 2),
                "Net Rev ($K)":     round(r["net_rev"] / 1000, 1),
            })

        df_out = pd.DataFrame(results_all)

        # Summary KPIs
        pepsi_rows = df_out[df_out["Brand"] == "PepsiCo"]
        total_vol_chg  = pepsi_rows["Vol Change (K)"].sum()
        total_rev_chg  = pepsi_rows["Rev Change ($K)"].sum()
        total_trade    = pepsi_rows["Trade Invest ($K)"].sum()
        total_net      = pepsi_rows["Net Rev ($K)"].sum()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"#### PepsiCo Portfolio Impact Summary")
        kk1, kk2, kk3, kk4 = st.columns(4)
        with kk1:
            vc = color_val(total_vol_chg)
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-label">Total PepsiCo Vol Change</div>
              <div class="kpi-value" style="color:{vc};">{total_vol_chg:+.1f}K</div>
              <div class="kpi-sub">units/week (PepsiCo SKUs only)</div>
            </div>""", unsafe_allow_html=True)
        with kk2:
            rc = color_val(total_rev_chg)
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-label">Revenue Impact</div>
              <div class="kpi-value" style="color:{rc};">${total_rev_chg:+.0f}K</div>
              <div class="kpi-sub">per week (PepsiCo)</div>
            </div>""", unsafe_allow_html=True)
        with kk3:
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-label">Trade Investment</div>
              <div class="kpi-value" style="color:{AMBER};">${total_trade:.1f}K</div>
              <div class="kpi-sub">per week (at ${trade_cost:.2f}/unit)</div>
            </div>""", unsafe_allow_html=True)
        with kk4:
            nc = color_val(total_net)
            st.markdown(f"""<div class="kpi-card">
              <div class="kpi-label">Net Revenue (Post-Trade)</div>
              <div class="kpi-value" style="color:{nc};">${total_net:+.0f}K</div>
              <div class="kpi-sub">per week</div>
            </div>""", unsafe_allow_html=True)

        # Waterfall chart
        st.markdown("<br>", unsafe_allow_html=True)
        c1w, c2w = st.columns([3, 2])
        with c1w:
            st.markdown("**📊 PepsiCo Volume Impact by SKU**")
            pepsi_df = df_out[df_out["Brand"] == "PepsiCo"].copy()
            fig5, ax5 = plt.subplots(figsize=(8, 4))
            fig5.patch.set_facecolor("#fafbfd")
            ax5.set_facecolor("#fafbfd")
            bar_colors = [GREEN if v >= 0 else RED for v in pepsi_df["Vol Change (K)"]]
            bars = ax5.bar(
                [l.replace(" 12oz×12", "\n12×12").replace(" 12oz×24", "\n12×24").replace(" 7.5oz×6", "\n7.5×6")
                 for l in pepsi_df["Product"]],
                pepsi_df["Vol Change (K)"],
                color=bar_colors, edgecolor="white", linewidth=0.5
            )
            for bar, val in zip(bars, pepsi_df["Vol Change (K)"]):
                ax5.text(bar.get_x() + bar.get_width()/2,
                         bar.get_height() + (0.1 if val >= 0 else -0.2),
                         f"{val:+.1f}K", ha="center", va="bottom", fontsize=8.5,
                         fontweight="bold", color="#333")
            ax5.axhline(0, color="#ccc", linewidth=0.8)
            ax5.set_ylabel("Volume Change (K units/week)", fontsize=9)
            ax5.set_title("Incremental Volume by PepsiCo SKU", fontsize=11,
                          fontweight="bold", color=NAVY)
            ax5.grid(axis="y", linestyle="--", alpha=0.3)
            ax5.spines[["top","right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig5, use_container_width=True)
            plt.close()

        with c2w:
            # Detail table
            st.markdown("**All Products — Detailed Impact**")
            display_df = df_out[["Product", "Brand", "Vol Change %", "Rev Change ($K)"]].copy()
            display_df.columns = ["Product", "Brand", "Vol %", "Rev ($K)"]
            st.dataframe(display_df, use_container_width=True, height=280,
                         hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — HOW TO USE
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        <div style="background:{NAVY}; color:white; padding:24px; border-radius:10px;">
          <div style="font-size:18px; font-weight:800; margin-bottom:14px;">📖 How to Use This Tool</div>

          <div style="font-size:13px; line-height:1.8;">

          <b>Step 1 — Load a Demo Scenario</b><br>
          Click any of the 3 buttons in the left panel to instantly populate
          all levers with a real recommended scenario.<br><br>

          <b>Step 2 — Explore Each Tab</b><br>
          • <b>Shelf Price (EDLP)</b> — model impact of regular price changes<br>
          • <b>Promotions (TPR)</b> — model impact of promotional discount events<br>
          • <b>Portfolio Summary</b> — see total impact across all SKUs at once<br><br>

          <b>Step 3 — Adjust the Levers</b><br>
          Move the sliders or type values to test any combination of
          price changes and promotional depths.<br><br>

          <b>Step 4 — Read the Business Outcome</b><br>
          The KPI cards instantly update with volume lift %, revenue impact,
          trade investment, and ROI.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown(f"""
        <div style="background:white; padding:24px; border-radius:10px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
          <div style="font-size:18px; font-weight:800; color:{NAVY}; margin-bottom:14px;">
            🔢 How the Numbers Work
          </div>
          <div style="font-size:13px; line-height:1.8; color:#444;">

          <b>EDLP Elasticity</b><br>
          Measures how sensitive shoppers are to regular shelf price.
          An elasticity of −2.43 means: if Pepsi's regular price goes up 1%,
          volume drops by about 2.43%. This is a "ceteris paribus" effect —
          it holds promotions, weather, and display constant.<br><br>

          <b>TPR Elasticity</b><br>
          Measures the volume lift from a promotional discount event.
          A value of 0.030 means each 1-percentage-point of discount depth
          drives ~3% more volume sold during that week.<br><br>

          <b>Cross-Price Effect</b><br>
          Shows the competitive relationship. Pepsi vs Coke cross-price = +1.35:
          when Coke raises its regular price by 1%, Pepsi gains about 1.35% extra volume
          — shoppers switch brands.<br><br>

          <b>Trade Investment & ROI</b><br>
          Trade cost is an estimate of the promotional spend per incremental unit.
          ROI = net revenue gain ÷ trade investment × 100.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:#f0f4ff; border: 1px solid {NAVY}30;
                padding:16px 20px; border-radius:8px; font-size:12px; color:#555;">
    <b>Data & Model Notes:</b>
    Model fitted on weekly store-level scanner data from ADUSA Corp-RMA – Grocery channel,
    Jan 2023 – Dec 2024. 8 CSD multipacks (Pepsi, Coca-Cola, Mountain Dew, Sprite).
    OLS estimation via statsmodels.OLS. Non-promoted (EDLP) baseline price derived via
    rolling 13-week maximum. Competition mapping follows FMCG same-category rule
    (Cola vs Cola, Citrus vs Citrus). R² range: 0.62–0.98. Results are indicative —
    week-level, single-store estimates. Validate directional findings before committing
    to pricing or trade spend decisions.
    </div>
    """, unsafe_allow_html=True)
