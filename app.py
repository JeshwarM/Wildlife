"""
Project Sentinel - Wildlife-Vehicle Collision Predictive System
Streamlit Dashboard: Dual-Auth (Citizen / Forest Authority)
Amber/Dark Theme | Folium Heatmap | SHAP Explainability | Future Forecasting
"""

import os
import warnings
import io
import json
import re
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
import shap

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR      = os.path.join(BASE_DIR, "models")
DATA_PATH       = os.path.join(MODELS_DIR, "preprocessed_data.csv")
HOTSPOT_PATH    = os.path.join(MODELS_DIR, "future_hotspots.csv")
SHAP_VAL_PATH   = os.path.join(MODELS_DIR, "shap_values.npy")
SHAP_FEAT_PATH  = os.path.join(MODELS_DIR, "shap_feature_names.npy")
CAT_PATH        = os.path.join(MODELS_DIR, "cat_expert.pkl")
XGB_PATH        = os.path.join(MODELS_DIR, "xgb_expert.pkl")
META_PATH       = os.path.join(MODELS_DIR, "meta_judge.pkl")
LE_PATH         = os.path.join(MODELS_DIR, "label_encoders.pkl")
FEEDBACK_PATH   = os.path.join(BASE_DIR, "officer_feedback.csv")

AI_DEFAULT_BASE_URL = "https://api.openai.com/v1/chat/completions"
AI_DEFAULT_MODEL = "gpt-4o-mini"
AI_ALLOWED_HOSTS_DEFAULT = "api.openai.com,localhost,127.0.0.1"
AI_MAX_REQUESTS_PER_SESSION = 40
AI_MIN_REQUEST_INTERVAL_SEC = 3
AI_MAX_PROMPT_CHARS = 4000
AI_MAX_CONTEXT_CHARS = 2500
AI_CHAT_HISTORY_WINDOW = 8

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project Sentinel",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# THEME INJECTION
# ─────────────────────────────────────────────────────────────
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --card: #ffffff;
  --ring: #f59e0b;
  --input: #e5e7eb;
  --muted: #f9fafb;
  --accent: #fffbeb;
  --border: #e5e7eb;
  --radius: 0.375rem;
  --chart-1: #f59e0b;
  --chart-2: #d97706;
  --chart-3: #b45309;
  --chart-4: #92400e;
  --chart-5: #78350f;
  --popover: #ffffff;
  --primary: #f59e0b;
  --sidebar: #f9fafb;
  --secondary: #f3f4f6;
  --background: #ffffff;
  --foreground: #262626;
  --destructive: #ef4444;
  --sidebar-ring: #f59e0b;
  --sidebar-accent: #fffbeb;
  --sidebar-border: #e5e7eb;
  --card-foreground: #262626;
  --sidebar-primary: #f59e0b;
  --muted-foreground: #6b7280;
  --accent-foreground: #92400e;
  --primary-foreground: #000000;
  --sidebar-foreground: #262626;
  --secondary-foreground: #4b5563;
  --destructive-foreground: #ffffff;
  --sidebar-accent-foreground: #92400e;
  --sidebar-primary-foreground: #ffffff;
}

html, body, [data-testid="stAppViewContainer"] {
  font-family: 'Inter', sans-serif;
  background-color: var(--background) !important;
  color: var(--foreground) !important;
}

[data-testid="stSidebar"] {
  background-color: var(--sidebar) !important;
  border-right: 1px solid var(--sidebar-border) !important;
}

[data-testid="stSidebar"] * {
  color: var(--sidebar-foreground) !important;
}

h1, h2, h3, h4, h5, h6 {
  font-family: 'Inter', sans-serif;
  font-weight: 600;
  color: var(--foreground);
}

/* Metric cards */
[data-testid="metric-container"] {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem;
  box-shadow: 0px 4px 8px rgba(0,0,0,0.08);
}

/* Primary button */
.stButton > button {
  background-color: var(--primary) !important;
  color: var(--primary-foreground) !important;
  border: none !important;
  border-radius: var(--radius) !important;
  font-family: 'Inter', sans-serif !important;
  font-weight: 500 !important;
  padding: 0.5rem 1.25rem !important;
  transition: background-color 0.2s ease;
}
.stButton > button:hover {
  background-color: var(--chart-2) !important;
}

/* Input fields */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stSlider {
  border-radius: var(--radius) !important;
  border-color: var(--border) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  gap: 0.25rem;
  border-bottom: 2px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  background-color: transparent;
  border-radius: var(--radius) var(--radius) 0 0;
  font-family: 'Inter', sans-serif;
  font-weight: 500;
  color: var(--muted-foreground);
  padding: 0.5rem 1rem;
  border: 1px solid transparent;
}
.stTabs [aria-selected="true"] {
  background-color: var(--accent) !important;
  color: var(--accent-foreground) !important;
  border-color: var(--border) !important;
  border-bottom-color: var(--accent) !important;
}

/* Divider */
hr { border-color: var(--border); }

/* Sentinel banner */
.sentinel-banner {
  background: linear-gradient(135deg, #f59e0b 0%, #92400e 100%);
  border-radius: 0.75rem;
  padding: 2rem 2.5rem;
  color: #000;
  margin-bottom: 1.5rem;
}
.sentinel-banner h1 {
  font-size: 2rem;
  font-weight: 700;
  color: #000;
  margin: 0 0 0.25rem 0;
}
.sentinel-banner p {
  font-size: 0.95rem;
  color: #1a1a1a;
  margin: 0;
}

/* KPI card */
.kpi-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.25rem 1.5rem;
  box-shadow: 0px 4px 8px rgba(0,0,0,0.07);
  text-align: center;
  transition: box-shadow 0.2s;
}
.kpi-card:hover {
  box-shadow: 0px 6px 14px rgba(245,158,11,0.2);
}
.kpi-label {
  font-size: 0.78rem;
  font-weight: 500;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--muted-foreground);
  margin-bottom: 0.5rem;
}
.kpi-value {
  font-size: 1.9rem;
  font-weight: 700;
  color: var(--primary);
}
.kpi-sub {
  font-size: 0.75rem;
  color: var(--muted-foreground);
  margin-top: 0.25rem;
}

/* Danger index bar */
.danger-bar-outer {
  background: var(--input);
  border-radius: 9999px;
  height: 14px;
  width: 100%;
  overflow: hidden;
  margin-top: 0.5rem;
}
.danger-bar-inner {
  height: 100%;
  border-radius: 9999px;
  background: linear-gradient(90deg, #f59e0b, #ef4444);
  transition: width 0.4s ease;
}

/* Login card */
.login-card {
  max-width: 420px;
  margin: 4rem auto;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 0.75rem;
  padding: 2.5rem;
  box-shadow: 0px 8px 24px rgba(0,0,0,0.1);
}
.login-card h2 {
  text-align: center;
  color: var(--primary);
  margin-bottom: 0.25rem;
}
.login-card p {
  text-align: center;
  color: var(--muted-foreground);
  font-size: 0.87rem;
  margin-bottom: 1.5rem;
}

/* Info alert */
.info-alert {
  background: var(--accent);
  border-left: 4px solid var(--primary);
  border-radius: var(--radius);
  padding: 0.75rem 1rem;
  color: var(--accent-foreground);
  font-size: 0.87rem;
  margin-bottom: 1rem;
}

/* Hotspot badge */
.hotspot-badge {
  display: inline-block;
  background: #fee2e2;
  color: #991b1b;
  border-radius: 9999px;
  padding: 0.15rem 0.6rem;
  font-size: 0.72rem;
  font-weight: 600;
}
.safe-badge {
  display: inline-block;
  background: #d1fae5;
  color: #065f46;
  border-radius: 9999px;
  padding: 0.15rem 0.6rem;
  font-size: 0.72rem;
  font-weight: 600;
}

/* Scrollable table */
.scroll-table {
  max-height: 320px;
  overflow-y: auto;
  border: 1px solid var(--border);
  border-radius: var(--radius);
}

code {
  font-family: 'JetBrains Mono', monospace;
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────
if "role" not in st.session_state:
    st.session_state.role = None
if "username" not in st.session_state:
    st.session_state.username = ""
if "ai_toolbar_enabled" not in st.session_state:
    st.session_state.ai_toolbar_enabled = False
if "ai_consent_given" not in st.session_state:
    st.session_state.ai_consent_given = False
if "ai_chat_history" not in st.session_state:
    st.session_state.ai_chat_history = []
if "ai_requests_count" not in st.session_state:
    st.session_state.ai_requests_count = 0
if "ai_last_request_ts" not in st.session_state:
    st.session_state.ai_last_request_ts = 0.0

CREDENTIALS = {
    "citizen":   ("sentinel123",  "citizen"),
    "authority": ("forestadmin", "authority"),
}


# ─────────────────────────────────────────────────────────────
# DATA LOADING (CACHED)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def load_models():
    models = {}
    try:
        models["cat"]  = joblib.load(CAT_PATH)
        models["xgb"]  = joblib.load(XGB_PATH)
        models["meta"] = joblib.load(META_PATH)
        models["le"]   = joblib.load(LE_PATH)
    except FileNotFoundError:
        pass
    return models


@st.cache_data(show_spinner="Loading data...")
def load_data_cached():
    if not os.path.exists(DATA_PATH):
        return pd.DataFrame()
    df = pd.read_csv(DATA_PATH, low_memory=False)
    return df


@st.cache_data(show_spinner=False)
def load_hotspots():
    if not os.path.exists(HOTSPOT_PATH):
        return pd.DataFrame()
    return pd.read_csv(HOTSPOT_PATH, low_memory=False)


@st.cache_data(show_spinner=False)
def load_shap():
    if not os.path.exists(SHAP_VAL_PATH) or not os.path.exists(SHAP_FEAT_PATH):
        return None, None
    shap_values = np.load(SHAP_VAL_PATH, allow_pickle=True)
    feat_names  = np.load(SHAP_FEAT_PATH, allow_pickle=True).tolist()
    return shap_values, feat_names


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def danger_index_html(pct: float) -> str:
    color = "#f59e0b" if pct < 50 else "#ef4444"
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">Overall Danger Index</div>
      <div class="kpi-value" style="color:{color};">{pct:.1f}%</div>
      <div class="danger-bar-outer">
        <div class="danger-bar-inner" style="width:{pct}%; background: linear-gradient(90deg,#f59e0b,{color});"></div>
      </div>
      <div class="kpi-sub">Based on AI risk score (max threshold: 10 incidents)</div>
    </div>
    """


def kpi_html(label: str, value: str, sub: str = "") -> str:
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {'<div class="kpi-sub">' + sub + '</div>' if sub else ''}
    </div>
    """


def build_heatmap(df: pd.DataFrame, risk_col: str = "danger_index") -> folium.Map:
    from folium.plugins import HeatMap
    if df.empty or "lat" not in df.columns or "lon" not in df.columns:
        return folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")

    # Prevent browser freezes and blank maps by capping the number of plotted points
    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42)

    center_lat = df["lat"].median()
    center_lon = df["lon"].median()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
        tiles="CartoDB positron",
    )

    heat_col = risk_col if risk_col in df.columns else "count"
    heat_data = df[["lat", "lon", heat_col]].dropna().values.tolist()
    heat_data = [[r[0], r[1], float(r[2])] for r in heat_data if -90 <= r[0] <= 90 and -180 <= r[1] <= 180]

    HeatMap(
        heat_data,
        min_opacity=0.3,
        radius=15,
        blur=12,
        gradient={0.2: "#78350f", 0.5: "#d97706", 0.8: "#f59e0b", 1.0: "#ef4444"},
    ).add_to(m)

    return m


def month_label(m: int) -> str:
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    return months[m - 1] if 1 <= m <= 12 else str(m)


def sanitize_text(value: str) -> str:
    if not value:
        return ""
    redacted = re.sub(r"(?i)(password|passwd|secret|token|api[_-]?key)\s*[:=]\s*\S+", r"\1=[REDACTED]", value)
    redacted = re.sub(r"(?i)bearer\s+[a-z0-9\-\._~\+\/]+=*", "Bearer [REDACTED]", redacted)
    return redacted.strip()


def build_ai_context(current_page: str, filtered: pd.DataFrame, include_schema: bool,
                     include_filters: bool, manual_context: str) -> str:
    context = {"page": current_page}

    if include_filters:
        context["active_filters"] = {
            "month": month_label(int(st.session_state.get("month_filter", 6))),
            "animal_class": st.session_state.get("animal_filter", "All"),
            "road_type": st.session_state.get("road_filter", "All"),
        }

    if not filtered.empty:
        context["filtered_rows"] = int(len(filtered))
        if "danger_index" in filtered.columns:
            context["avg_danger_index"] = round(float(filtered["danger_index"].mean()), 3)
        if "count" in filtered.columns:
            context["total_incidents_filtered"] = int(filtered["count"].sum())

    if include_schema and not filtered.empty:
        schema = []
        for col in filtered.columns[:20]:
            dtype = str(filtered[col].dtype)
            non_null = int(filtered[col].notna().sum())
            schema.append({"name": col, "dtype": dtype, "non_null": non_null})
        context["data_schema"] = schema

    if manual_context:
        context["user_context"] = sanitize_text(manual_context[:AI_MAX_CONTEXT_CHARS])

    context_json = json.dumps(context, ensure_ascii=False)
    return context_json[:AI_MAX_CONTEXT_CHARS]


def is_allowed_ai_host(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    allow_raw = os.getenv("AI_ALLOWED_HOSTS", AI_ALLOWED_HOSTS_DEFAULT)
    allowed_hosts = {h.strip().lower() for h in allow_raw.split(",") if h.strip()}
    return host in allowed_hosts


def call_llm(messages: list, max_tokens: int, temperature: float) -> tuple[bool, str]:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    base_url = os.getenv("OPENAI_BASE_URL", AI_DEFAULT_BASE_URL).strip()
    model = os.getenv("OPENAI_MODEL", AI_DEFAULT_MODEL).strip()

    if not api_key:
        return (
            False,
            "AI backend is not configured. Set OPENAI_API_KEY (and optional OPENAI_BASE_URL / OPENAI_MODEL) to enable live responses.",
        )
    if not is_allowed_ai_host(base_url):
        return False, "Configured AI endpoint is not on the allowed host list."

    body = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    payload = json.dumps(body).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    req = Request(base_url, data=payload, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=30) as resp:
            response_data = json.loads(resp.read().decode("utf-8"))
            choices = response_data.get("choices", [])
            if not choices:
                return False, "AI provider returned no response choices."
            content = choices[0].get("message", {}).get("content", "").strip()
            if not content:
                return False, "AI response was empty."
            return True, content
    except HTTPError as e:
        return False, f"AI request failed: HTTP {e.code}"
    except URLError:
        return False, "AI request failed: network error."
    except Exception as e:
        return False, f"AI request failed: unexpected error ({type(e).__name__})."


def stream_chunks(text: str):
    for chunk in text.split():
        yield chunk + " "


def render_ai_toolbar(current_page: str, filtered: pd.DataFrame):
    st.markdown("### AI Toolbar")
    st.session_state.ai_toolbar_enabled = st.toggle(
        "Enable assistant",
        value=st.session_state.ai_toolbar_enabled,
        help="Toggle AI assistant visibility and interactions.",
    )

    if not st.session_state.ai_toolbar_enabled:
        st.caption("Turn on the assistant to ask questions about current dashboard data.")
        return

    with st.expander("Assistant Panel", expanded=True):
        st.caption("Privacy: The assistant only uses context you explicitly include.")
        st.session_state.ai_consent_given = st.checkbox(
            "I consent to sending selected context to the configured AI provider",
            value=st.session_state.ai_consent_given,
        )

        c1, c2 = st.columns(2)
        include_filters = c1.checkbox("Include filters", value=True)
        include_schema = c2.checkbox("Include dataset schema", value=False)

        max_tokens = st.slider("Response token cap", min_value=128, max_value=800, value=350, step=32)
        temperature = st.slider("Creativity", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        manual_context = st.text_area(
            "Optional extra context",
            height=90,
            placeholder="Add relevant notes (no passwords, tokens, or credentials).",
        )

        if st.button("Clear chat", use_container_width=True):
            st.session_state.ai_chat_history = []
            st.success("Chat cleared.")

        for msg in st.session_state.ai_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        prompt = st.chat_input("Ask the AI assistant")
        if prompt:
            if not st.session_state.ai_consent_given:
                st.error("Consent is required before sending any context.")
                return

            now = time.time()
            if st.session_state.ai_requests_count >= AI_MAX_REQUESTS_PER_SESSION:
                st.warning("Session rate limit reached. Clear chat or restart session.")
                return
            if now - st.session_state.ai_last_request_ts < AI_MIN_REQUEST_INTERVAL_SEC:
                st.warning("Please wait a few seconds before sending another request.")
                return

            safe_prompt = sanitize_text(prompt[:AI_MAX_PROMPT_CHARS]).strip()
            context_json = build_ai_context(
                current_page=current_page,
                filtered=filtered,
                include_schema=include_schema,
                include_filters=include_filters,
                manual_context=manual_context,
            )

            system_msg = (
                "You are a wildlife risk analysis assistant. "
                "Use only the provided context and ask for clarification when missing data."
            )
            user_msg = f"Context JSON:\n{context_json}\n\nUser question:\n{safe_prompt}"
            messages = [{"role": "system", "content": system_msg}] + st.session_state.ai_chat_history[-AI_CHAT_HISTORY_WINDOW:] + [
                {"role": "user", "content": user_msg}
            ]

            st.session_state.ai_chat_history.append({"role": "user", "content": safe_prompt})
            st.session_state.ai_last_request_ts = now
            st.session_state.ai_requests_count += 1

            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    ok, response_text = call_llm(messages, max_tokens=max_tokens, temperature=temperature)
                if ok:
                    st.write_stream(stream_chunks(response_text))
                    st.session_state.ai_chat_history.append({"role": "assistant", "content": response_text.strip()})
                else:
                    fallback = (
                        f"{response_text}\n\n"
                        "Fallback guidance: Review current KPIs, heatmap, and selected filters, "
                        "then ask a focused question like 'summarize top risks this month'."
                    )
                    st.warning(fallback)
                    st.session_state.ai_chat_history.append({"role": "assistant", "content": fallback})


# ─────────────────────────────────────────────────────────────
# LOGIN PAGE
# ─────────────────────────────────────────────────────────────
def render_login():
    st.markdown(
        '<div class="login-card">'
        '<h2>Project Sentinel</h2>'
        '<p>Wildlife-Vehicle Collision Predictive System<br>'
        'Authorized access only. Select your role and enter credentials.</p>',
        unsafe_allow_html=True,
    )
    username = st.text_input("Username", placeholder="citizen or authority", key="login_user")
    password = st.text_input("Password", type="password", placeholder="Enter password", key="login_pass")

    if st.button("Sign In", use_container_width=True):
        u = username.strip().lower()
        if u in CREDENTIALS:
            correct_pw, role = CREDENTIALS[u]
            if password == correct_pw:
                st.session_state.role     = role
                st.session_state.username = u
                st.rerun()
            else:
                st.error("Incorrect password.")
        else:
            st.error("Username not recognised. Use 'citizen' or 'authority'.")

    st.markdown(
        '<div class="info-alert" style="margin-top:1.5rem;">Citizen login: '
        '<code>citizen / sentinel123</code><br>'
        'Authority login: <code>authority / forestadmin</code></div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
def render_sidebar(df: pd.DataFrame):
    with st.sidebar:
        st.markdown(
            '<div style="padding:0.75rem 0 1.25rem;">'
            '<span style="font-size:1.2rem;font-weight:700;color:var(--primary);">Project Sentinel</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        role_label = "Forest Authority" if st.session_state.role == "authority" else "Citizen"
        st.markdown(
            f'<div class="info-alert">Signed in as: <strong>{st.session_state.username}</strong>'
            f'<br>Role: <strong>{role_label}</strong></div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("**Filters**")

        month_val = st.slider(
            "Month",
            min_value=1,
            max_value=12,
            value=6,
            format="%d",
            help="Filter incidents by calendar month",
        )
        st.caption(f"Selected: {month_label(month_val)}")

        animal_options = ["All"]
        if not df.empty and "class" in df.columns:
            animal_options += sorted(df["class"].dropna().unique().tolist())
        animal_class = st.selectbox("Animal Class", options=animal_options)

        road_options = ["All"]
        if not df.empty and "roadType" in df.columns:
            road_options += sorted(df["roadType"].dropna().unique().tolist())
        road_type = st.selectbox("Road Type", options=road_options)

        st.session_state.month_filter = month_val
        st.session_state.animal_filter = animal_class
        st.session_state.road_filter = road_type

        st.markdown("---")
        if st.button("Sign Out", use_container_width=True):
            st.session_state.role     = None
            st.session_state.username = ""
            st.rerun()

    return month_val, animal_class, road_type


# ─────────────────────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────────────────────
def apply_filters(df: pd.DataFrame, month_val: int, animal_class: str, road_type: str) -> pd.DataFrame:
    if df.empty:
        return df
    filt = df.copy()
    if "month" in filt.columns:
        filt = filt[filt["month"] == month_val]
    if animal_class != "All" and "class" in filt.columns:
        filt = filt[filt["class"] == animal_class]
    if road_type != "All" and "roadType" in filt.columns:
        filt = filt[filt["roadType"] == road_type]
    return filt


# ─────────────────────────────────────────────────────────────
# CITIZEN VIEW
# ─────────────────────────────────────────────────────────────
def render_citizen(df: pd.DataFrame, filtered: pd.DataFrame, models: dict):
    st.markdown(
        '<div class="sentinel-banner">'
        '<h1>Project Sentinel</h1>'
        '<p>Wildlife-Vehicle Collision Risk Intelligence | Real-Time Predictive Dashboard</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # KPI row
    total_incidents = int(df["count"].sum()) if not df.empty and "count" in df.columns else 0
    unique_species  = df["class"].nunique()  if not df.empty and "class" in df.columns else 0
    road_segments   = df["roadType"].nunique() if not df.empty and "roadType" in df.columns else 0
    avg_danger      = float(filtered["danger_index"].mean()) if not df.empty and "danger_index" in df.columns and not filtered.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_html("Total Incidents",     f"{total_incidents:,}", "Global records"), unsafe_allow_html=True)
    c2.markdown(kpi_html("Animal Species",      str(unique_species),    "Distinct classes"), unsafe_allow_html=True)
    c3.markdown(kpi_html("Road Segment Types",  str(road_segments),     "In dataset"), unsafe_allow_html=True)
    c4.markdown(danger_index_html(avg_danger), unsafe_allow_html=True)

    st.markdown("---")

    # Heatmap
    st.markdown("### Incident Heatmap")
    st.caption("Spatial distribution of wildlife-vehicle collisions. Intensity driven by AI risk score.")

    if filtered.empty:
        st.info("No records match the current filters. Adjust the sidebar controls.")
    else:
        m = build_heatmap(filtered, risk_col="danger_index")
        st_folium(m, use_container_width=True, height=520, returned_objects=[])

    # Monthly trend chart
    if not df.empty and "month" in df.columns and "count" in df.columns:
        st.markdown("### Monthly Incident Trend")
        monthly = df.groupby("month")["count"].sum().reset_index()
        monthly["month_label"] = monthly["month"].apply(month_label)
        fig = go.Figure(go.Bar(
            x=monthly["month_label"],
            y=monthly["count"],
            marker_color="#f59e0b",
        ))
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            font_family="Inter",
            xaxis_title="Month",
            yaxis_title="Total Incidents",
            margin=dict(l=0, r=0, t=20, b=0),
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# AUTHORITY TABS
# ─────────────────────────────────────────────────────────────
def render_shap_tab(shap_values, feat_names, df: pd.DataFrame):
    if shap_values is None or feat_names is None:
        st.warning("SHAP artifacts not found. Run train_pipeline.py first.")
        return

    st.markdown("### Global Feature Importance (Mean |SHAP|)")
    mean_abs = np.abs(shap_values).mean(axis=0)
    shap_df  = pd.DataFrame({"Feature": feat_names, "Mean |SHAP|": mean_abs})
    shap_df  = shap_df.sort_values("Mean |SHAP|", ascending=True)

    fig = go.Figure(go.Bar(
        x=shap_df["Mean |SHAP|"],
        y=shap_df["Feature"],
        orientation="h",
        marker=dict(
            color=shap_df["Mean |SHAP|"],
            colorscale=[[0, "#f9fafb"], [0.5, "#f59e0b"], [1, "#92400e"]],
            showscale=False,
        ),
    ))
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_family="Inter",
        xaxis_title="Mean Absolute SHAP Value",
        margin=dict(l=0, r=0, t=10, b=0),
        height=max(280, 40 * len(feat_names)),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### SHAP Value Distribution (Beeswarm-style Scatter)")
    fig2 = go.Figure()
    for i, feat in enumerate(feat_names):
        vals = shap_values[:, i]
        fig2.add_trace(go.Scatter(
            x=vals,
            y=[feat] * len(vals),
            mode="markers",
            marker=dict(size=3, color=vals, colorscale="RdYlGn", opacity=0.4),
            name=feat,
            showlegend=False,
        ))
    fig2.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_family="Inter",
        xaxis_title="SHAP Value (impact on prediction)",
        height=max(300, 40 * len(feat_names)),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig2, use_container_width=True)


def render_forecast_tab(hotspots: pd.DataFrame, df: pd.DataFrame):
    st.markdown("### Future Hotspot Forecast")
    st.markdown(
        '<div class="info-alert">Segments with historically low incident counts but high predicted '
        'future risk scores. These zones are candidates for preventative infrastructure investment.</div>',
        unsafe_allow_html=True,
    )

    if hotspots.empty:
        st.warning("No future hotspots file found. Run train_pipeline.py to generate forecasts.")
        return

    # Summary metrics
    h1, h2, h3 = st.columns(3)
    h1.markdown(kpi_html("Hotspot Segments",  str(len(hotspots)),           "Flagged by AI"), unsafe_allow_html=True)
    avg_risk = hotspots["future_risk_score"].mean() if "future_risk_score" in hotspots.columns else 0
    h2.markdown(kpi_html("Avg Future Risk Score", f"{avg_risk:.2f}",         "/ 10.0 max"), unsafe_allow_html=True)
    avg_danger = hotspots["danger_index"].mean() if "danger_index" in hotspots.columns else 0
    h3.markdown(kpi_html("Avg Danger Index",  f"{avg_danger:.1f}%",          "Normalized"), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Hotspot Map (6-month Forward Projection)")
    m = build_heatmap(hotspots, risk_col="future_risk_score")
    # Add circle markers for top hotspots
    if "lat" in hotspots.columns and "lon" in hotspots.columns:
        top_n = hotspots.head(20)
        for _, row in top_n.iterrows():
            try:
                risk = float(row.get("future_risk_score", 5))
                popup_text = (
                    f"Risk Score: {risk:.2f}<br>"
                    f"Class: {row.get('class','Unknown')}<br>"
                    f"Road: {row.get('roadType','Unknown')}"
                )
                folium.CircleMarker(
                    location=[float(row["lat"]), float(row["lon"])],
                    radius=6,
                    color="#ef4444",
                    fill=True,
                    fill_color="#ef4444",
                    fill_opacity=0.8,
                    popup=folium.Popup(popup_text, max_width=200),
                ).add_to(m)
            except Exception:
                pass
    st_folium(m, use_container_width=True, height=480, returned_objects=[])

    st.markdown("---")
    st.markdown("### Top 20 Identified Future Hotspots")
    display_cols = [c for c in ["lat","lon","class","roadType","count","future_risk_score","danger_index"] if c in hotspots.columns]
    st.dataframe(
        hotspots[display_cols].head(20).reset_index(drop=True),
        use_container_width=True,
        height=300,
    )


def render_segment_explorer(df: pd.DataFrame, shap_values, feat_names):
    st.markdown("### Segment Explorer")
    st.caption("Select a row index to drill into a single road segment and view its local SHAP explanation.")

    if df.empty:
        st.warning("No data loaded.")
        return

    display_cols = [c for c in ["lat","lon","class","roadType","month","count","danger_index"] if c in df.columns]
    sample = df[display_cols].dropna(subset=["lat","lon"] if "lat" in display_cols else []).head(200)

    idx = st.number_input(
        "Segment Row Index (0-indexed from table below)",
        min_value=0,
        max_value=max(0, len(sample) - 1),
        value=0,
        step=1,
    )

    st.dataframe(sample.reset_index(drop=True), use_container_width=True, height=280)

    if shap_values is not None and feat_names is not None:
        if idx < len(shap_values):
            st.markdown("---")
            st.markdown(f"#### Local SHAP Explanation - Row {idx}")
            row_shap = shap_values[idx]
            local_df = pd.DataFrame({
                "Feature": feat_names,
                "SHAP Value": row_shap,
            }).sort_values("SHAP Value", key=abs, ascending=True)

            colors = ["#f59e0b" if v >= 0 else "#ef4444" for v in local_df["SHAP Value"]]
            fig = go.Figure(go.Bar(
                x=local_df["SHAP Value"],
                y=local_df["Feature"],
                orientation="h",
                marker_color=colors,
            ))
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                font_family="Inter",
                xaxis_title="SHAP Value",
                xaxis=dict(zeroline=True, zerolinecolor="#e5e7eb"),
                margin=dict(l=0, r=0, t=10, b=0),
                height=max(250, 38 * len(feat_names)),
            )
            st.plotly_chart(fig, use_container_width=True)
            positive_feats = local_df[local_df["SHAP Value"] > 0]["Feature"].tolist()
            negative_feats = local_df[local_df["SHAP Value"] < 0]["Feature"].tolist()
            if positive_feats:
                st.markdown(
                    f'**Risk-increasing factors:** {", ".join(positive_feats[-3:])}'
                )
            if negative_feats:
                st.markdown(
                    f'**Risk-reducing factors:** {", ".join(negative_feats[:3])}'
                )
        else:
            st.info("Row index exceeds available SHAP rows. Run training on the full dataset for full coverage.")


def render_feedback_tab():
    st.markdown("### Officer Field Feedback Log")
    st.caption("Submit observations from field officers. Submissions are appended to a local CSV for audit and future model retraining.")

    with st.form("feedback_form", clear_on_submit=True):
        f1, f2 = st.columns(2)
        officer_name = f1.text_input("Officer Name")
        segment_id   = f2.text_input("Segment / Location ID")

        f3, f4 = st.columns(2)
        obs_date   = f3.date_input("Observation Date")
        severity   = f4.selectbox("Incident Severity", ["Low", "Moderate", "High", "Critical"])

        notes = st.text_area("Field Notes", height=100, placeholder="Describe the observed conditions...")

        submitted = st.form_submit_button("Submit Feedback", use_container_width=True)
        if submitted:
            row = {
                "officer":    officer_name,
                "segment":    segment_id,
                "date":       str(obs_date),
                "severity":   severity,
                "notes":      notes,
            }
            fb_df = pd.DataFrame([row])
            if os.path.exists(FEEDBACK_PATH):
                existing = pd.read_csv(FEEDBACK_PATH)
                fb_df = pd.concat([existing, fb_df], ignore_index=True)
            fb_df.to_csv(FEEDBACK_PATH, index=False)
            st.success("Feedback submitted successfully.")

    if os.path.exists(FEEDBACK_PATH):
        st.markdown("---")
        st.markdown("**Logged Submissions**")
        fb = pd.read_csv(FEEDBACK_PATH)
        st.dataframe(fb, use_container_width=True, height=260)


def render_bulk_upload(df: pd.DataFrame):
    st.markdown("---")
    st.markdown("### Bulk Data Upload")
    st.caption("Upload a CSV to extend the dataset. The file must contain at least `lat`, `lon`, `class`, and `roadType` columns.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            new_df = pd.read_csv(uploaded, encoding="utf-8-sig")
            st.success(f"Uploaded {len(new_df):,} rows.")
            st.dataframe(new_df.head(10), use_container_width=True)
            st.info("To incorporate this data into the model, save it alongside the original CSV and re-run train_pipeline.py.")
        except Exception as e:
            st.error(f"Error reading file: {e}")


def render_authority(df: pd.DataFrame, filtered: pd.DataFrame, hotspots: pd.DataFrame,
                     shap_values, feat_names, models: dict):
    st.markdown(
        '<div class="sentinel-banner">'
        '<h1>Project Sentinel - Authority Operations Portal</h1>'
        '<p>Forest Authority | Full Analytical Access | Explainable AI Dashboard</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Same citizen KPIs + heatmap at top
    total_incidents = int(df["count"].sum())        if not df.empty and "count" in df.columns else 0
    unique_species  = df["class"].nunique()          if not df.empty and "class" in df.columns else 0
    road_segments   = df["roadType"].nunique()       if not df.empty and "roadType" in df.columns else 0
    avg_danger      = float(filtered["danger_index"].mean()) if not df.empty and "danger_index" in df.columns and not filtered.empty else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi_html("Total Incidents",    f"{total_incidents:,}", "Global records"), unsafe_allow_html=True)
    c2.markdown(kpi_html("Animal Species",     str(unique_species),    "Distinct classes"), unsafe_allow_html=True)
    c3.markdown(kpi_html("Road Segment Types", str(road_segments),     "In dataset"), unsafe_allow_html=True)
    c4.markdown(danger_index_html(avg_danger), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Live Incident Heatmap")
    if not filtered.empty:
        m = build_heatmap(filtered, risk_col="danger_index")
        st_folium(m, use_container_width=True, height=460, returned_objects=[])
    else:
        st.info("No records match the current filters.")

    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs([
        "Feature Importance (SHAP)",
        "Forecast Panel",
        "Segment Explorer",
        "Officer Feedback",
    ])

    with tab1:
        render_shap_tab(shap_values, feat_names, df)

    with tab2:
        render_forecast_tab(hotspots, df)

    with tab3:
        render_segment_explorer(df, shap_values, feat_names)

    with tab4:
        render_feedback_tab()

    render_bulk_upload(df)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    # Load resources
    models      = load_models()
    df          = load_data_cached()
    hotspots    = load_hotspots()
    shap_values, feat_names = load_shap()

    # Auth check
    if st.session_state.role is None:
        render_login()
        return

    # Sidebar filters
    month_val, animal_class, road_type = render_sidebar(df)
    filtered = apply_filters(df, month_val, animal_class, road_type)

    # Route by role
    content_col, ai_col = st.columns([4, 1.35], gap="large")

    with content_col:
        if st.session_state.role == "citizen":
            render_citizen(df, filtered, models)
            current_page = "Citizen Dashboard"
        else:
            render_authority(df, filtered, hotspots, shap_values, feat_names, models)
            current_page = "Authority Operations Portal"

    with ai_col:
        render_ai_toolbar(current_page=current_page, filtered=filtered)


if __name__ == "__main__":
    main()
