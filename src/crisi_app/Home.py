import os 
import streamlit as st
import requests
from pathlib import Path
DEFAULT_API_URL = os.getenv("CRISI_BACKEND_URL", st.secrets.get("backend_url", ""))
API_URL = DEFAULT_API_URL.rstrip("/")


LOGO_PATH = Path(r"uaegean_logo.png")

st.set_page_config(
    page_title="CRISI v2 — UAegean",
    page_icon=str(LOGO_PATH) if LOGO_PATH.exists() else "🌍",
    layout="wide",
)

# --- Header with logo + title + author ---
col_logo, col_text = st.columns([1, 6], vertical_alignment="center")

with col_logo:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=90)

with col_text:
    st.title("CRISI v2: Climate Resilience Investment Scoring Tool")
    st.caption(
        "University of the Aegean — PhD Thesis Prototype | "
        "Resilience and Sustainability of Tourism Investments in the Era of Climate Change | "
        "Creator: Vangelis Zaftis"
    )

@st.cache_data(show_spinner=False)
def check_api_health(api_url: str):
    """Ping the FastAPI backend and fetch scenarios if possible."""
    health = None
    scenarios = None
    error = None

    try:
        r = requests.get(f"{api_url}/health", timeout=3)
        r.raise_for_status()
        health = r.json()
    except Exception as e:
        error = f"Health check failed: {e}"
        return health, scenarios, error

    try:
        r = requests.get(f"{api_url}/scenarios", timeout=5)
        r.raise_for_status()
        scenarios = r.json()
    except Exception as e:
        # Backend is up, but scenarios endpoint may have an issue
        error = f"Could not load scenarios: {e}"

    return health, scenarios, error



st.markdown(
    """
CRISI v2 is a prototype **climate-resilient investment scoring and CBA tool** for tourism
projects under multiple climate–socioeconomic scenarios.

The app is split into pages (see the sidebar) that let you:

- Explore **resilience trajectories** by region and scenario  
- Compare **scenarios** for the same region  
- Run **CBA for a single project** under one scenario  
- Compare **CBA across scenarios** for the same project  
- Compare **with vs without adaptation** for a project under one scenario  
"""
)

# --------------------------------------------------------------------
# API status panel
# --------------------------------------------------------------------
st.subheader("Backend status")

health, scenarios, error = check_api_health(API_URL)

status_col, info_col = st.columns([1, 2])

with status_col:
    if health and health.get("status") == "ok":
        st.success("FastAPI backend is **running**.")
    else:
        st.error("FastAPI backend is **not reachable**.")
        if error:
            st.write(error)

with info_col:
    if scenarios:
        st.markdown("**Available scenarios:**")
        for s in scenarios:
            st.markdown(
                f"- `{s['id']}` – **{s['name']}**  \n"
                f"&nbsp;&nbsp;&nbsp;&nbsp;{s.get('description','')}"
            )
    else:
        st.info(
            "Scenarios could not be listed. If you just started the backend, "
            "try refreshing the page."
        )

st.write("---")

# --------------------------------------------------------------------
# How to use the app
# --------------------------------------------------------------------
st.subheader("How to use this app")

st.markdown(
    """
### 1. Resilience & scenarios

- **01 – Resilience Explorer**  
  Choose a region and a scenario and compute time series of:
  - exposure (heat, drought),
  - sensitivity (tourism dependence, arrivals, income vulnerability),
  - adaptive capacity (education, health, governance),
  - and the combined **resilience index**.

  You can switch between:
  - **Policy (fixed weights):** uses the default configuration (e.g. Delphi-based).
  - **Sandbox (adjustable weights):** you can change pillar weights and sub-weights to see
    how the resilience trajectory reacts.

- **02 – Scenario Comparison**  
  Compare resilience paths for the **same region** across several scenarios side by side.
"""
)

st.markdown(
    """
### 2. Project CBA under climate uncertainty

- **03 – Project CBA**  
  Evaluate a single tourism investment under one scenario.  
  You can configure:
  - CAPEX, construction period, depreciation, salvage value  
  - Revenue and OPEX dynamics, tax rate  
  - Adaptation share of CAPEX and subsidies  
  - Strength of the link between resilience and revenues  

  You can run it with:
  - **Policy defaults** for resilience weights, or  
  - **Custom (sandbox)** weights for sensitivity analysis.

- **04 – Project CBA – Scenario Comparison**  
  Hold project parameters fixed, change only the **scenario**, and compare:
  - NPV  
  - IRR  
  - Payback year  
  - Cumulative discounted cashflows  
  across all selected scenarios.

- **05 – Project CBA – With vs Without Adaptation**  
  Compare two variants of the same project under **one scenario**:
  - Case A: **no adaptation** (adaptation share ≈ 0, no subsidy)  
  - Case B: **with adaptation** (adaptation CAPEX + subsidy)  

  The page reports ΔNPV, ΔIRR, Δsubsidy and a simple *“subsidy efficiency”* metric (ΔNPV / Δsubsidy).
"""
)

st.write("---")

# --------------------------------------------------------------------
# Weights & modes explanation
# --------------------------------------------------------------------
st.subheader("Weights, modes and interpretation")

st.markdown(
    """
**Pillar weights** determine how much each component contributes to the resilience index:

- Exposure (climate hazard pressure)
- Sensitivity (how exposed the tourism system is socio-economically)
- Adaptive capacity (how well the system can cope and adjust)

Each pillar is further broken into **subcomponents** (e.g. heat vs drought, tourism dependence vs arrivals, etc.).

There are two modes:

- **Policy defaults**  
  - Uses the internal default configuration (e.g. Delphi/expert-based).  
  - Intended for *official* or *reference* evaluations.

- **Custom (sandbox)**  
  - Lets you change pillar and sub-weights freely.  
  - Intended for **sensitivity analysis** and exploring how assumptions affect results.  
  - All results produced in this mode are explicitly marked as sandbox in the UI and CSV exports.
"""
)

st.write("---")

# --------------------------------------------------------------------
# Data & limitations
# --------------------------------------------------------------------
st.subheader("Current status & limitations")

st.markdown(
    """
This is a **prototype**:

- Hazard and socio-economic data may currently be **synthetic or partial** in this version.  
- Scenario narratives (e.g. Green Global Resilience, Techno-Optimism on a Hot Planet, Regional Fortress)
  are implemented as stylised hazard/resilience paths and will need to be tied to
  **IPCC / SSP–RCP data** in later stages.
- All outputs are for **exploratory analysis** and method testing – not yet for formal
  investment decisions.

When you export CSVs from any page, the files include:
- Region, scenario(s)
- Key financial and project assumptions
- Weights mode and, for sandbox runs, the full set of custom weights
"""
)

st.info(
    "Use the sidebar on the left to navigate between pages. "
    "Start with **Resilience Explorer** to get a feel for the trajectories, "
    "then move to **Project CBA** and the comparison pages."
)
