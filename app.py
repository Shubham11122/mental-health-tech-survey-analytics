"""
app.py
------
Mental Health in Tech Survey — interactive Streamlit application.

Run:
    streamlit run app.py

Expects cleaned_survey_readable.csv in the same folder.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =============================================================
# PAGE CONFIG + CUSTOM STYLE
# =============================================================
st.set_page_config(
    page_title="Mental Health in Tech — Survey Analytics",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

PRIMARY = "#2F6F6B"      # single accent — muted teal
INK = "#1F2328"          # near-black text
MUTED = "#6E7781"        # secondary text / labels
GREY = "#9AA5B1"         # the one other color, for binary contrast
BG = "#FFFFFF"           # plain white background
SURFACE = "#F4F5F6"      # subtle off-white for cards

# No CSS overrides on Streamlit's own widgets (sidebar, selectbox,
# expander, tabs, buttons) here. Every earlier attempt to force
# color on those broke something else — popups render in portals
# outside the elements they're nested under in the markup, some
# labels use inline styles CSS can't reach cleanly, and the result
# was a mix of black-on-black in different spots each time. Streamlit
# already guarantees correct contrast for all of its own components
# *within whichever theme is active* — so the only thing that needs
# fixing is making sure the intended light theme is the one that's
# actually active, which the run command below does directly rather
# than depending on a config file being found.
CUSTOM_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* KPI cards — fully self-authored HTML, explicit colors on both
   background and text, so this can't drift out of sync with
   whatever theme is active. */
.kpi-card {{
    background-color: {SURFACE};
    border: 1px solid #E5E7EB;
    border-left: 3px solid {PRIMARY};
    border-radius: 6px;
    padding: 14px 18px;
}}
.kpi-label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: {MUTED};
    margin-bottom: 2px;
}}
.kpi-value {{
    font-size: 1.7rem;
    font-weight: 700;
    color: {INK};
}}

/* Harmless hover lift on buttons — adds motion only, touches no
   color property, so it can't conflict with theme colors either
   way. */
[data-testid^="stBaseButton"] {{
    border-radius: 6px !important;
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}}
[data-testid^="stBaseButton"]:hover {{
    transform: translateY(-1px);
    box-shadow: 0 3px 10px rgba(0,0,0,0.10);
}}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

TREATMENT_COLORS = {"Yes": PRIMARY, "No": GREY}
CATEGORICAL_PALETTE = [PRIMARY, "#5C948F", GREY, "#C7CDD3", "#1C4441"]
DIVERGING_SCALE = [[0, GREY], [0.5, "#FFFFFF"], [1, PRIMARY]]

DATA_PATH = "cleaned_survey_readable.csv"

EMPLOYEE_ORDER = ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
INTERFERE_ORDER = ["Not Applicable", "Never", "Rarely", "Sometimes", "Often"]
LEAVE_ORDER = ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"]


# =============================================================
# DATA LOADING
# =============================================================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    df["No Employees"] = pd.Categorical(df["No Employees"], categories=EMPLOYEE_ORDER, ordered=True)
    df["Work Interfere"] = pd.Categorical(df["Work Interfere"], categories=INTERFERE_ORDER, ordered=True)
    df["Leave"] = pd.Categorical(df["Leave"], categories=LEAVE_ORDER, ordered=True)

    df["Age Group"] = pd.cut(
        df["Age"], bins=[17, 24, 34, 44, 54, 75],
        labels=["18-24", "25-34", "35-44", "45-54", "55+"],
    )

    top_countries = df["Country"].value_counts().nlargest(8).index
    df["Country Grouped"] = df["Country"].where(df["Country"].isin(top_countries), "Other")

    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    """Quick logistic regression: predict Treatment from key workplace/demographic factors."""
    feature_cols = [
        "Age", "Gender", "Family History", "Work Interfere", "No Employees",
        "Remote Work", "Benefits", "Care Options", "Anonymity", "Leave", "Supervisor",
    ]
    model_df = df[feature_cols + ["Treatment"]].dropna().copy()

    encoders = {}
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(model_df[col]):
            le = LabelEncoder()
            model_df[col] = le.fit_transform(model_df[col].astype(str))
            encoders[col] = le

    X = model_df[feature_cols]
    y = (model_df["Treatment"] == "Yes").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)

    coef_df = pd.DataFrame({"Feature": feature_cols, "Coefficient": clf.coef_[0]})
    coef_df = coef_df.sort_values("Coefficient", key=abs, ascending=False)

    return clf, encoders, feature_cols, accuracy, coef_df


df_full = load_data()
model, encoders, feature_cols, model_accuracy, coef_df = train_model(df_full)


def kpi_card(label, value):
    st.markdown(
        f"""<div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
            </div>""",
        unsafe_allow_html=True,
    )

def show_chart(fig):
    """Render a Plotly figure with a guaranteed light appearance,
    regardless of the app's active light/dark theme setting — passing
    theme=None stops Streamlit from auto-reskinning the figure to
    match the active theme (which is what turned every chart black)."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font_color=INK,
        title_font_color=PRIMARY,
    )
    st.plotly_chart(fig, use_container_width=True, theme=None)



# =============================================================
# SIDEBAR
# =============================================================
with st.sidebar:
    st.markdown("### 🧭 Survey Filters")
    st.caption("OSMI Mental Health in Tech Survey · 2014")
    st.markdown("---")

    countries = sorted(df_full["Country"].unique().tolist())
    sel_countries = st.multiselect("Country", countries, default=[])

    genders = sorted(df_full["Gender"].unique().tolist())
    sel_genders = st.multiselect("Gender", genders, default=[])

    age_min, age_max = int(df_full["Age"].min()), int(df_full["Age"].max())
    sel_age = st.slider("Age range", age_min, age_max, (age_min, age_max))

    company_sizes = st.multiselect("Company size", EMPLOYEE_ORDER, default=[])

    tech_only = st.checkbox("Tech companies only", value=False)

    st.markdown("---")
    if st.button("↺ Reset all filters", use_container_width=True, type="primary"):
        st.rerun()

df = df_full.copy()
if sel_countries:
    df = df[df["Country"].isin(sel_countries)]
if sel_genders:
    df = df[df["Gender"].isin(sel_genders)]
df = df[df["Age"].between(sel_age[0], sel_age[1])]
if company_sizes:
    df = df[df["No Employees"].isin(company_sizes)]
if tech_only:
    df = df[df["Tech Company"] == "Yes"]

with st.sidebar:
    st.markdown(f"**{len(df):,}** of **{len(df_full):,}** respondents match your filters")

if len(df) == 0:
    st.warning("No respondents match the current filters. Adjust the sidebar and try again.")
    st.stop()

# =============================================================
# HEADER
# =============================================================
st.title("Mental Health in Tech Survey")
st.caption(
    "An interactive analysis of how workplace policy, culture, and personal "
    "history relate to whether tech employees seek mental health treatment."
)
st.divider()

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    kpi_card("Respondents", f"{len(df):,}")
with k2:
    kpi_card("Sought Treatment", f"{(df['Treatment'] == 'Yes').mean() * 100:.1f}%")
with k3:
    kpi_card("Family History", f"{(df['Family History'] == 'Yes').mean() * 100:.1f}%")
with k4:
    kpi_card("Median Age", f"{int(df['Age'].median())}")
with k5:
    kpi_card("Model Accuracy", f"{model_accuracy * 100:.0f}%")

st.write("")

# =============================================================
# TABS
# =============================================================
tab_overview, tab_workplace, tab_corr, tab_predict, tab_data, tab_insights = st.tabs(
    ["📊 Overview", "🏢 Workplace Factors", "🔗 Correlation Explorer",
     "🤖 Risk Predictor", "🗂️ Data Explorer", "💡 Insights"]
)

# -------------------------------------------------------------
# TAB 1: OVERVIEW
# -------------------------------------------------------------
with tab_overview:
    with st.expander("📋 About This Project — Problem Statement, Objective & Workflow", expanded=True):
        st.markdown("""
#### Problem Statement
Mental health remains under-discussed in tech workplaces, and many companies don't
know whether their policies (benefits, anonymity protections, leave policy,
manager/coworker support) actually correlate with employees seeking treatment.
This project analyzes the 2014 OSMI Mental Health in Tech Survey to identify which
workplace and demographic factors are most associated with an employee seeking
treatment for a mental health condition.

#### Business Objective
Identify the factors most strongly associated with treatment-seeking, so an
organization can prioritize the policies most likely to increase help-seeking and
reduce the stigma tied to disclosing a mental health condition at work.

#### Analysis Workflow
1. **Data Cleaning** — fixed `Age` outliers (invalid/impossible values), normalized
   49 raw `Gender` spellings into consistent categories, filled meaningful defaults
   for missing `state`/`self_employed`/`work_interfere`, removed duplicate
   responses → 1,255 clean records.
2. **Exploratory Data Analysis** — 20 visuals across demographics and every major
   workplace factor (family history, benefits, anonymity, leave, supervisor/coworker
   comfort, etc.) compared against treatment-seeking.
3. **Modeling** — a logistic regression trained on 11 workplace/demographic factors
   to surface which drive treatment-seeking most (see the **Risk Predictor** tab).
4. **This Dashboard** — turns the static analysis into something you can filter,
   explore, and query interactively.
        """)

    c1, c2 = st.columns([1.3, 1])
    with c1:
        fig = px.histogram(df, x="Age", nbins=20, title="Age Distribution",
                            color_discrete_sequence=[PRIMARY])
        fig.update_layout(bargap=0.05)
        show_chart(fig)
    with c2:
        counts = df["Treatment"].value_counts().reset_index()
        counts.columns = ["Treatment", "Count"]
        fig = px.pie(counts, names="Treatment", values="Count", hole=0.55,
                     title="Sought Treatment?",
                     color="Treatment", color_discrete_map=TREATMENT_COLORS)
        fig.update_traces(textinfo="percent+label")
        show_chart(fig)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.bar(df["Gender"].value_counts().reset_index(), x="Gender", y="count",
                     title="Gender Distribution", labels={"count": "Respondents"},
                     color="Gender", color_discrete_sequence=CATEGORICAL_PALETTE)
        show_chart(fig)
    with c4:
        order = df["Country Grouped"].value_counts().index
        fig = px.bar(df["Country Grouped"].value_counts().reindex(order).reset_index(),
                     x="count", y="Country Grouped", orientation="h",
                     title="Respondents by Country", labels={"count": "Respondents", "Country Grouped": "Country"},
                     color_discrete_sequence=[GREY])
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        show_chart(fig)

    st.subheader("Age Group vs Treatment Rate")
    rate = df.groupby("Age Group", observed=True)["Treatment"].apply(
        lambda s: (s == "Yes").mean() * 100
    ).reset_index(name="Treatment Rate")
    fig = px.line(rate, x="Age Group", y="Treatment Rate", markers=True,
                  title="Treatment-Seeking Rate Across Age Groups (%)",
                  color_discrete_sequence=[PRIMARY])
    fig.update_traces(line_width=3, marker_size=10)
    show_chart(fig)

# -------------------------------------------------------------
# TAB 2: WORKPLACE FACTORS (diverse chart types)
# -------------------------------------------------------------
with tab_workplace:
    st.caption("Each workplace factor rendered as the chart type that best fits its shape — "
               "hierarchies as sunbursts/treemaps, rates as bubbles, ordered scales as polar bars.")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        fig = px.sunburst(df, path=["Family History", "Treatment"],
                           title="Family History → Treatment",
                           color="Family History", color_discrete_sequence=CATEGORICAL_PALETTE)
        show_chart(fig)
    with r1c2:
        fig = px.treemap(df, path=["Benefits", "Treatment"],
                          title="Benefits → Treatment",
                          color="Benefits", color_discrete_sequence=CATEGORICAL_PALETTE)
        show_chart(fig)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        grp = df.groupby(["Work Interfere", "Treatment"], observed=True).size().reset_index(name="Count")
        fig = go.Figure()
        for t in ["Yes", "No"]:
            sub = grp[grp["Treatment"] == t].sort_values("Work Interfere")
            fig.add_trace(go.Barpolar(r=sub["Count"], theta=sub["Work Interfere"].astype(str),
                                       name=t, marker_color=TREATMENT_COLORS[t]))
        fig.update_layout(title="Work Interference vs Treatment (Polar)")
        show_chart(fig)
    with r2c2:
        rate = df.groupby("No Employees", observed=True).agg(
            Count=("Treatment", "size"),
            TreatmentRate=("Treatment", lambda s: (s == "Yes").mean() * 100),
        ).reset_index().sort_values("No Employees")
        fig = px.scatter(rate, x="No Employees", y="TreatmentRate", size="Count",
                          color="No Employees", size_max=55,
                          title="Company Size vs Treatment Rate (Bubble)",
                          labels={"TreatmentRate": "Treatment Rate (%)"},
                          color_discrete_sequence=CATEGORICAL_PALETTE)
        show_chart(fig)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        fig = px.sunburst(df, path=["Care Options", "Treatment"],
                           title="Care Options → Treatment",
                           color="Care Options", color_discrete_sequence=CATEGORICAL_PALETTE)
        show_chart(fig)
    with r3c2:
        rate = df.groupby("Anonymity").agg(
            Count=("Treatment", "size"),
            TreatmentRate=("Treatment", lambda s: (s == "Yes").mean() * 100),
        ).reset_index()
        fig = px.scatter(rate, x="Anonymity", y="TreatmentRate", size="Count",
                          color="Anonymity", size_max=55,
                          title="Anonymity vs Treatment Rate (Bubble)",
                          labels={"TreatmentRate": "Treatment Rate (%)"},
                          color_discrete_sequence=CATEGORICAL_PALETTE)
        show_chart(fig)

    r4c1, r4c2 = st.columns(2)
    with r4c1:
        grp = df.groupby(["Leave", "Treatment"], observed=True).size().reset_index(name="Count")
        fig = go.Figure()
        for t in ["Yes", "No"]:
            sub = grp[grp["Treatment"] == t].sort_values("Leave")
            fig.add_trace(go.Barpolar(r=sub["Count"], theta=sub["Leave"].astype(str),
                                       name=t, marker_color=TREATMENT_COLORS[t]))
        fig.update_layout(title="Leave Difficulty vs Treatment (Polar)")
        show_chart(fig)
    with r4c2:
        ct = pd.crosstab(df["Supervisor"], df["Treatment"], normalize="index") * 100
        fig = go.Figure()
        for t in ct.columns:
            fig.add_trace(go.Bar(name=t, x=ct.index, y=ct[t], marker_color=TREATMENT_COLORS[t]))
        fig.update_layout(barmode="stack", title="Supervisor Comfort vs Treatment (% Stacked)",
                           yaxis_title="Percentage")
        show_chart(fig)

    st.markdown("#### Explore any factor yourself")
    factor_options = {
        "Wellness Program": "Employer wellness program",
        "Seek Help": "Employer provides resources to seek help",
        "Remote Work": "Works remotely",
        "Self Employed": "Self-employed",
        "Coworkers": "Comfortable discussing with coworkers",
        "Mental Health Consequence": "Fear of negative consequences (mental)",
    }
    choice = st.selectbox("Pick a factor to compare against Treatment",
                           list(factor_options.keys()), format_func=lambda k: factor_options[k])
    fig = px.histogram(df, x=choice, color="Treatment", barmode="group",
                        title=f"{factor_options[choice]} vs Treatment",
                        color_discrete_map=TREATMENT_COLORS)
    show_chart(fig)

# -------------------------------------------------------------
# TAB 3: CORRELATION EXPLORER
# -------------------------------------------------------------
with tab_corr:
    st.subheader("Correlation Heatmap")
    encode_cols = [
        "Treatment", "Family History", "Work Interfere", "Remote Work",
        "Benefits", "Care Options", "Wellness Program", "Seek Help",
        "Anonymity", "Leave", "Mental Health Consequence",
        "Phys Health Consequence", "Supervisor", "Coworkers",
        "Obs Consequence", "Self Employed", "Age",
    ]
    encoded = df[encode_cols].copy()
    for col in encoded.columns:
        if not pd.api.types.is_numeric_dtype(encoded[col]):
            encoded[col] = LabelEncoder().fit_transform(encoded[col].astype(str))

    corr = encoded.corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale=DIVERGING_SCALE,
                     zmin=-1, zmax=1, title="Correlation Matrix (Label-Encoded)")
    fig.update_layout(height=650)
    show_chart(fig)

    st.markdown("#### Pair Plot")
    pairplot_cols = ["Age", "Family History", "Work Interfere", "Benefits", "Treatment"]
    pp_df = df[pairplot_cols].copy()
    for col in ["Family History", "Work Interfere", "Benefits"]:
        pp_df[col] = LabelEncoder().fit_transform(pp_df[col].astype(str))
    fig = px.scatter_matrix(pp_df, dimensions=["Age", "Family History", "Work Interfere", "Benefits"],
                             color="Treatment", color_discrete_map=TREATMENT_COLORS,
                             title="Pair Plot of Key Encoded Variables")
    fig.update_layout(height=650)
    show_chart(fig)

# -------------------------------------------------------------
# TAB 4: RISK PREDICTOR (real model-driven interactivity)
# -------------------------------------------------------------
with tab_predict:
    st.subheader("Treatment-Seeking Likelihood Estimator")
    st.caption(
        f"A logistic regression trained on {len(df_full):,} survey responses "
        f"(held-out accuracy: {model_accuracy * 100:.0f}%). Adjust the inputs to see how "
        f"the estimated likelihood of seeking treatment shifts."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        in_age = st.slider("Age", 18, 75, 30)
        in_gender = st.selectbox("Gender", sorted(df_full["Gender"].unique()))
        in_family = st.selectbox("Family history of mental illness", ["Yes", "No"])
    with c2:
        in_interfere = st.selectbox("Work interference", INTERFERE_ORDER, index=2)
        in_employees = st.selectbox("Company size", EMPLOYEE_ORDER, index=2)
        in_remote = st.selectbox("Works remotely", ["Yes", "No"])
    with c3:
        in_benefits = st.selectbox("Employer provides benefits", ["Yes", "No", "Don't know"])
        in_care = st.selectbox("Aware of care options", ["Yes", "No", "Not sure"])
        in_anonymity = st.selectbox("Anonymity protected", ["Yes", "No", "Don't know"])
    in_leave = st.select_slider("Ease of taking leave", options=LEAVE_ORDER, value="Don't know")
    in_supervisor = st.selectbox("Comfortable discussing with supervisor", sorted(df_full["Supervisor"].unique()))

    raw_input = {
        "Age": in_age, "Gender": in_gender, "Family History": in_family,
        "Work Interfere": in_interfere, "No Employees": in_employees,
        "Remote Work": in_remote, "Benefits": in_benefits, "Care Options": in_care,
        "Anonymity": in_anonymity, "Leave": in_leave, "Supervisor": in_supervisor,
    }

    encoded_input = []
    for col in feature_cols:
        val = raw_input[col]
        if col in encoders:
            le = encoders[col]
            val = le.transform([str(val)])[0] if str(val) in le.classes_ else 0
        encoded_input.append(val)

    proba = model.predict_proba([encoded_input])[0][1]

    st.markdown("---")
    g1, g2 = st.columns([1, 2])
    with g1:
        st.markdown(
            f"""<div class="kpi-card" style="text-align:center;">
                    <div class="kpi-label">Estimated Likelihood</div>
                    <div class="kpi-value" style="font-size:2.6rem;">{proba*100:.0f}%</div>
                    <div style="color:{MUTED}; font-size:0.85rem;">of seeking treatment</div>
                </div>""",
            unsafe_allow_html=True,
        )
    with g2:
        fig = px.bar(coef_df, x="Coefficient", y="Feature", orientation="h",
                     title="What Drives the Model (Logistic Regression Coefficients)",
                     color="Coefficient", color_continuous_scale=DIVERGING_SCALE,
                     color_continuous_midpoint=0)
        fig.update_layout(height=350)
        show_chart(fig)

    st.info(
        "This is a simple statistical estimate for exploration, not a diagnostic or "
        "clinical tool — treatment-seeking is a personal decision shaped by many "
        "factors this small model can't capture."
    )

# -------------------------------------------------------------
# TAB 5: DATA EXPLORER
# -------------------------------------------------------------
with tab_data:
    st.subheader("Filtered Raw Data")
    st.caption("This reflects the sidebar filters currently applied.")
    search_col = st.selectbox("Search within column (optional)", ["(none)"] + df.columns.tolist())
    view_df = df.copy()
    if search_col != "(none)":
        term = st.text_input(f"Search '{search_col}' contains...")
        if term:
            view_df = view_df[view_df[search_col].astype(str).str.contains(term, case=False, na=False)]

    st.dataframe(view_df, use_container_width=True, height=420)
    st.download_button(
        "⬇️ Download filtered data as CSV",
        data=view_df.to_csv(index=False).encode("utf-8"),
        file_name="mental_health_filtered.csv",
        mime="text/csv",
        type="primary",
    )

# -------------------------------------------------------------
# TAB 6: INSIGHTS
# -------------------------------------------------------------
with tab_insights:
    st.subheader("Key Findings")
    st.markdown("""
- **Family history** and **self-reported work interference** are the strongest individual
  correlates of treatment-seeking — confirmed both in the charts and as the top two
  coefficients in the predictive model.
- **Awareness gaps matter as much as policy gaps.** "Don't know" responses for `Benefits`,
  `Care Options`, and `Anonymity` behave much like "No" responses — many employees likely
  have support they don't know about.
- **Remote work, company size, age, and gender** show weak standalone relationships with
  treatment-seeking — none are reliable levers on their own.
- The model reaches roughly **{:.0f}% accuracy** on held-out data using only 11 workplace
  and demographic factors — good enough to highlight drivers, not to make individual
  decisions from.
    """.format(model_accuracy * 100))

    st.subheader("Recommendations")
    st.markdown("""
1. Communicate existing benefits, care options, and anonymity guarantees clearly and repeatedly.
2. Train managers to recognize early signs of work interference.
3. Use voluntary, anonymous outreach informed by family-history risk signals.
4. Extend support to self-employed and small-company workers who fall outside standard benefit structures.
5. Avoid relying on remote-work policy or company size alone as a mental-health lever.
    """)

st.markdown("---")
st.caption("Data: OSMI Mental Health in Tech Survey, 2014 (Kaggle) · Built with Streamlit & Plotly")
