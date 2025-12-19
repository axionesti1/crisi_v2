import streamlit as st
from pathlib import Path
from crisi_core.config import DEFAULT_SCENARIOS


LOGO_PATH = Path(r"C:\Users\USER\OneDrive\Desktop\crisi_v2\uaegean_logo.png")

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

#
# In the original version of this page the app pinged a FastAPI backend to
# check its status and list scenarios.  This refactored prototype has no
# dependency on an external API: the Streamlit app imports the core
# resilience and CBA functions directly and performs all computations
# in‑memory.  See the scenario narratives section below for the list of
# supported scenarios.



st.markdown(
    """
CRISI v2 is a prototype **climate‑resilient investment scoring and cost–benefit analysis (CBA) tool** for
tourism projects under multiple climate–socioeconomic scenarios.

Unlike earlier versions, this Streamlit app does **not** rely on a separate FastAPI backend;
instead, it imports the core resilience and CBA functions directly.  All calculations run on the
server hosting this app, so there is no external API to start or monitor.

Use the sidebar to navigate between pages.  The pages let you:

- Explore **resilience trajectories** for a selected region and scenario.  
- Compare **scenarios** for the same region.  
- Perform **side‑by‑side comparisons** of regions and scenarios on a dashboard with maps and explainability.  
- Run **CBA for a single project** under one or many scenarios.  
- Compare **with vs without adaptation** for a project under a given scenario.  
- Experiment with **AI/XAI surrogates** for resilience and CBA to see which inputs drive the results.
"""
)

# --------------------------------------------------------------------
# Scenario narratives
# --------------------------------------------------------------------
st.subheader("Scenario narratives")

st.markdown(
    """
CRISI v2 currently supports five stylised scenarios that span optimistic,
middle‑of‑the‑road and pessimistic futures.  Each scenario combines a
climate pathway (e.g. RCP4.5 vs RCP8.5) with a socio‑economic story (e.g.
SSP1 vs SSP5).  The resilience and hazard trajectories in the app are
synthetic and are meant to illustrate how different futures might unfold.
"""
)

for sid, s in DEFAULT_SCENARIOS.items():
    st.markdown(
        f"- `{sid}` – **{s.name}**  \n"
        f"&nbsp;&nbsp;&nbsp;&nbsp;{s.description}"
    )

st.write("---")

# --------------------------------------------------------------------
# How to use the app
# --------------------------------------------------------------------
st.subheader("How to use this app")

st.markdown(
    """
### 1. Resilience & scenarios

- **Resilience Explorer**  
  Select a region and a scenario to generate time‑series of **exposure** (heat, drought and, where available, flood risk), **sensitivity** (tourism dependence, arrivals and income vulnerability), **adaptive capacity** (education, health and governance) and the combined **resilience index**.  
  Two modes are available:  
  - **Policy defaults** use internally defined weights (e.g. expert‑based) for the pillars and subcomponents.  
  - **Sandbox** lets you adjust weights and introduce wildcards to see how different assumptions affect the trajectory.

- **Scenario Comparison**  
  View resilience trajectories for a single region across multiple scenarios on one chart.

- **Comparison Dashboard**  
  A multi‑tab dashboard to:  
  - Compare **two regions** under the same scenario, including exposure/sensitivity/adaptive components and XAI summaries.  
  - Compare **one region** across all scenarios using any metric (resilience, risk, exposure, sensitivity, adaptive).  
  - Visualise values on a **map** of Greek NUTS‑2 regions with a colour legend.
"""
)

st.markdown(
    """
### 2. Project CBA under climate uncertainty

‑ **Project CBA Dashboard**  
  Analyse a tourism investment’s financial performance under climate uncertainty.  The dashboard has three tabs:  
  - **Resilience preview** shows the region’s resilience and risk trajectories for the chosen scenario and lets you adjust weights.  
  - **CBA across scenarios** allows you to fix project parameters and compute **net present value (NPV)** under all scenarios.  It also offers explainable‑AI panels that decompose NPV drivers and provide a tornado sensitivity chart.  
  - **Adaptation vs no‑adaptation** compares two variants of the project (with and without adaptation CAPEX/subsidy) for one scenario, reporting ΔNPV and visualising cashflow differences.

‑ **AI/XAI Lab**  
  A laboratory page that uses surrogate models (trained on synthetic CRISI runs) to approximate the resilience engine and CBA.  It provides **SHAP‑based explanations** for individual cases and grouped attributions, enabling rapid insight into which inputs push predictions up or down.  Use it for exploration and intuition, then validate conclusions on the core pages.
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
