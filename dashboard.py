"""
INDIA LIFE EXPECTANCY DASHBOARD — REAL DATA VERSION
WHO + World Bank + NFHS-5 Data
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json, os, warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="India Life Expectancy — XAI", page_icon="🫀",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Source+Sans+3:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'Source Sans 3', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif; }
[data-testid="stSidebar"] { background: linear-gradient(180deg,#0a0a1a,#0d1b2a); }
[data-testid="stSidebar"] * { color: #e0e8f0 !important; }
.metric-card { background:linear-gradient(135deg,#0d1b2a,#1a3a5c); border:1px solid #2a5f8f;
  border-radius:12px; padding:20px; text-align:center; color:white; }
.metric-value { font-size:2.2em; font-weight:700; color:#4fc3f7; font-family:'Playfair Display',serif; }
.metric-label { font-size:0.85em; color:#94b8d4; text-transform:uppercase; letter-spacing:1px; margin-top:4px; }
.hero-banner { background:linear-gradient(135deg,#0a0a1a,#0d2137,#0a3d62); padding:40px 30px;
  border-radius:16px; margin-bottom:24px; border:1px solid #1e5f8a; }
.hero-title { font-family:'Playfair Display',serif; font-size:2.4em; color:#fff; margin:0; line-height:1.2; }
.hero-subtitle { color:#94b8d4; font-size:1.05em; margin-top:10px; }
.hero-badge { display:inline-block; background:rgba(79,195,247,0.15); border:1px solid #4fc3f7;
  color:#4fc3f7; padding:4px 14px; border-radius:20px; font-size:0.8em; font-weight:600;
  letter-spacing:1px; text-transform:uppercase; margin-bottom:16px; }
.section-header { border-left:4px solid #4fc3f7; padding-left:14px; margin:30px 0 16px; }
.section-header h2 { color:#1a3a5c; margin:0; font-size:1.6em; }
            .js-plotly-plot .plotly .legend text { fill: #1a3a5c !important; }
.section-header p  { color:#5a7a9a; margin:4px 0 0; font-size:0.9em; }
.insight-box { background:#f0f7ff; border:1px solid #bee3f8; border-radius:10px;
  padding:16px 20px; margin:10px 0; border-left:4px solid #4fc3f7; color:#1a3a5c; }
.prediction-result { background:linear-gradient(135deg,#0d2137,#0a3d62); color:white;
  padding:24px; border-radius:14px; text-align:center; border:1px solid #2a7aaf; }
.prediction-number { font-size:3.5em; font-weight:700; color:#4fc3f7; font-family:'Playfair Display',serif; }
</style>
""", unsafe_allow_html=True)

# ── LOAD DATA ──
@st.cache_data
def load_data():
    who     = pd.read_csv('data/who_cleaned.csv')
    states  = pd.read_csv('data/india_states.csv')
    trend   = pd.read_csv('data/india_trend.csv')
    future  = pd.read_csv('data/future_predictions.csv')
    shap_df = pd.read_csv('data/shap_importance.csv')
    global_ = pd.read_csv('data/global_data.csv')
    avp     = pd.read_csv('data/actual_vs_predicted.csv')
    with open('data/metrics.json')     as f: metrics = json.load(f)
    with open('data/causal_edges.json') as f: causal = json.load(f)
    return who, states, trend, future, shap_df, global_, avp, metrics, causal

who, states, trend, future, shap_df, global_df, avp, metrics, causal_edges = load_data()

@st.cache_resource
def load_model():
    from ml_engine import load_who_data, train_model
    df = load_who_data()
    model, rf, features, X_test, y_test, y_pred, rf_pred, r2, mae, rf_r2, rf_mae = train_model(df)
    medians = df.select_dtypes(include=[np.number]).median().to_dict()
    return model, rf, features, medians

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("## 🫀 Navigation")
    page = st.radio("", [
        "🏠  Overview",
        "🌍  Global Comparison",
        "🗺️  India State Analysis",
        "🔮  Future Predictions",
        "🔍  XAI — SHAP",
        "🔗  Causal Modelling",
        "🧮  Personal Predictor",
        "📊  Model Comparison",
        "📜  Ancestral Insights"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.markdown(f"**XGBoost R²:** `{metrics['xgb_r2']}`")
    st.markdown(f"**XGBoost Acc:** `{metrics['xgb_accuracy_pct']}%`")
    st.markdown(f"**MAE:** `{metrics['xgb_mae']} years`")
    st.markdown(f"**Data:** `{metrics['n_samples']} real records`")
    st.markdown(f"**Source:** WHO + World Bank")
    st.markdown("---")
    st.caption("India Life Expectancy\nXAI + Causal Modelling")

# ═══════════════════════════════
# PAGE 1: OVERVIEW
# ═══════════════════════════════
if "Overview" in page:
    st.markdown("""
    <div class="hero-banner">
        <h1 class="hero-title">Predicting & Extending<br>Human Life Expectancy in India</h1>
        <p class="hero-subtitle">Explainable AI · Causal Modelling · 193 Countries · 25 Indian States</p>
    </div>""", unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(val,lbl) in zip([c1,c2,c3,c4,c5],[
        ("72.2 yrs","India (2024)"),("85.1 yrs","World Best"),
        ("54.0 yrs","World Lowest"),(f"{metrics['xgb_accuracy_pct']}%","Model Accuracy"),
        (f"{metrics['n_samples']:,}","Real Records")
    ]):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                     f'<div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([3,2])

    with col1:
        st.markdown('<div class="section-header"><h2>India\'s Life Expectancy — 1960 to 2024</h2>'
                    '<p>Real World Bank data</p></div>', unsafe_allow_html=True)
        fig = px.area(trend, x='year', y='life_expectancy', color_discrete_sequence=['#4fc3f7'])
        fig.update_layout(plot_bgcolor='#f8fbff', paper_bgcolor='white',
                          xaxis_title="Year", yaxis_title="Life Expectancy (Years)",
                          yaxis=dict(range=[35,80]), margin=dict(l=10,r=10,t=10,b=10),
                          font=dict(family='Source Sans 3'), showlegend=False)
        fig.add_annotation(x=2021, y=67.3, text="<b>COVID dip (2021)</b>",
                           showarrow=True, arrowhead=2, font=dict(color='#e57373'))
        fig.add_annotation(x=2024, y=72.2, text="<b>72.2 yrs (2024)</b>",
                           showarrow=True, arrowhead=2, font=dict(color='#1a3a5c'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header"><h2>Why This Matters</h2></div>', unsafe_allow_html=True)
        for txt in [
            "🔴 <b>14+ year gap</b> between India and top nations like Japan",
            "🏥 India has only <b>0.7 doctors per 1000</b> vs 2.5+ in developed nations",
            "🌫️ <b>13 of 20</b> most polluted cities globally are in India",
            "📉 <b>Mortality before 50</b> has risen significantly in past decades",
            "✅ <b>XAI + Causal AI</b> to explain and predict — not just model"
        ]:
            st.markdown(f'<div class="insight-box">{txt}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header"><h2>Methodology</h2></div>', unsafe_allow_html=True)
    for col,(num,title,desc) in zip(st.columns(5),[
        ("1️⃣","Real Data","WHO + World Bank + NFHS-5"),
        ("2️⃣","Feature Eng.","22 health & socioeconomic vars"),
        ("3️⃣","ML Models","XGBoost + Random Forest"),
        ("4️⃣","XAI (SHAP)","Feature importance & explanations"),
        ("5️⃣","Causal AI","DoWhy causal effect estimation"),
    ]):
        col.markdown(f'<div class="metric-card"><div style="font-size:2em">{num}</div>'
                     f'<div style="font-size:0.95em;font-weight:700;color:#e0f0ff;margin:8px 0 4px">{title}</div>'
                     f'<div style="font-size:0.78em;color:#94b8d4">{desc}</div></div>', unsafe_allow_html=True)


# ═══════════════════════════════
# PAGE 2: GLOBAL COMPARISON
# ═══════════════════════════════
elif "Global" in page:
    st.markdown('<div class="section-header"><h2>🌍 Global Life Expectancy Comparison</h2>'
                '<p>Real World Bank data — latest available year per country</p></div>', unsafe_allow_html=True)

    colors = {'Asia':'#4fc3f7','Europe':'#81c784','N. America':'#ffb74d',
              'S. America':'#f06292','Africa':'#e57373','Oceania':'#ce93d8'}

    fig = px.bar(global_df.sort_values('life_expectancy'),
                 x='life_expectancy', y='country', color='continent',
                 orientation='h', color_discrete_map=colors, text='life_expectancy')
    fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
    fig.add_vline(x=72.2, line_dash='dash', line_color='#ff7043',
                  annotation_text='India 72.2', annotation_position='top right')
    fig.update_layout(height=600, plot_bgcolor='#f8fbff', paper_bgcolor='white',
                      xaxis_title="Life Expectancy (Years)", yaxis_title="",
                      font=dict(family='Source Sans 3'), margin=dict(l=10,r=60,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

    c1,c2,c3 = st.columns(3)
    c1.markdown('<div class="insight-box"><b>🇯🇵 Japan (~84 yrs)</b><br>Universal healthcare, low obesity, traditional fish-vegetable diet, strong community bonds.</div>', unsafe_allow_html=True)
    c2.markdown('<div class="insight-box"><b>🇮🇳 India (72.2 yrs)</b><br>Gap of 12+ years vs Japan. Pollution, healthcare access, nutrition, and stress are key drivers.</div>', unsafe_allow_html=True)
    c3.markdown('<div class="insight-box"><b>🇳🇬 Nigeria / Chad (~54 yrs)</b><br>Poverty, infectious disease, and lack of healthcare access drive the world\'s lowest numbers.</div>', unsafe_allow_html=True)


# ═══════════════════════════════
# PAGE 3: INDIA STATE ANALYSIS
# ═══════════════════════════════
elif "India State" in page:
    st.markdown('<div class="section-header"><h2>🗺️ India State-wise Life Expectancy</h2>'
                '<p>Based on SRS 2018-22 estimates + NFHS-5 health indicators</p></div>', unsafe_allow_html=True)

    sorted_states = states.sort_values('life_expectancy', ascending=False)

    col1, col2 = st.columns([2,1])
    with col1:
        fig = px.bar(sorted_states, x='state', y='life_expectancy',
                     color='life_expectancy', color_continuous_scale='RdYlGn',
                     text='life_expectancy')
        fig.update_traces(texttemplate='%{text:.1f}', textposition='outside')
        fig.update_layout(height=430, plot_bgcolor='#f8fbff', paper_bgcolor='white',
                          xaxis_tickangle=-45, font=dict(family='Source Sans 3'),
                          coloraxis_showscale=False, xaxis_title="", yaxis_title="Life Expectancy (yrs)",
                          margin=dict(l=10,r=10,t=20,b=90))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**🟢 Top 5 States**")
        for _,row in sorted_states.head(5).iterrows():
            st.markdown(f"🟢 **{row['state']}** — {row['life_expectancy']} yrs")
        st.markdown("---")
        st.markdown("**🔴 Bottom 5 States**")
        for _,row in sorted_states.tail(5).iterrows():
            st.markdown(f"🔴 **{row['state']}** — {row['life_expectancy']} yrs")
        gap = sorted_states['life_expectancy'].max() - sorted_states['life_expectancy'].min()
        st.markdown(f"---\n**State Gap:** `{gap:.1f} years`")

    st.markdown('<div class="section-header"><h2>Key NFHS-5 Indicators by State</h2>'
                '<p>Select an indicator to compare across states</p></div>', unsafe_allow_html=True)

    indicator = st.selectbox("Choose indicator", {
        'sanitation_pct': '🚽 Sanitation Access (%)',
        'clean_fuel_pct': '🔥 Clean Cooking Fuel (%)',
        'health_insurance_pct': '🏥 Health Insurance Coverage (%)',
        'vaccination_pct': '💉 Children Fully Vaccinated (%)',
        'tobacco_use_pct': '🚬 Tobacco Use Among Men (%)',
        'women_literacy_pct': '📚 Women Literacy (%)',
        'anaemia_women_pct': '🩸 Anaemia Among Women (%)',
        'doctors_per_1000': '👨‍⚕️ Doctors per 1000 Population'
    }.keys(), format_func=lambda x: {
        'sanitation_pct': '🚽 Sanitation Access (%)',
        'clean_fuel_pct': '🔥 Clean Cooking Fuel (%)',
        'health_insurance_pct': '🏥 Health Insurance Coverage (%)',
        'vaccination_pct': '💉 Children Fully Vaccinated (%)',
        'tobacco_use_pct': '🚬 Tobacco Use Among Men (%)',
        'women_literacy_pct': '📚 Women Literacy (%)',
        'anaemia_women_pct': '🩸 Anaemia Among Women (%)',
        'doctors_per_1000': '👨‍⚕️ Doctors per 1000 Population'
    }[x])

    fig2 = px.scatter(states, x=indicator, y='life_expectancy', text='state',
                      color='life_expectancy', color_continuous_scale='RdYlGn', size_max=20,
                      size=[15]*len(states))
    fig2.update_traces(textposition='top center', textfont_size=9)
    fig2.update_layout(plot_bgcolor='#f8fbff', paper_bgcolor='white',
                       font=dict(family='Source Sans 3'), coloraxis_showscale=False,
                       yaxis_title="Life Expectancy (yrs)", margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig2, use_container_width=True)


# ═══════════════════════════════
# PAGE 4: FUTURE PREDICTIONS
# ═══════════════════════════════
elif "Future" in page:
    st.markdown('<div class="section-header"><h2>🔮 India Future Life Expectancy (2024–2050)</h2>'
                '<p>XGBoost predictions under 3 policy scenarios</p></div>', unsafe_allow_html=True)

    fig = go.Figure()
    sc_colors = {'Business as Usual':'#ffb74d',
                 'Optimistic (Policy Reforms)':'#81c784',
                 'Pessimistic (No Action)':'#e57373'}
    sc_dash   = {'Business as Usual':'dash',
                 'Optimistic (Policy Reforms)':'solid',
                 'Pessimistic (No Action)':'dot'}
    for sc, color in sc_colors.items():
        fig.add_trace(go.Scatter(x=future['year'], y=future[sc], name=sc,
                                 line=dict(color=color, width=3, dash=sc_dash[sc]), mode='lines'))
    fig.add_trace(go.Scatter(
        x=list(future['year'])+list(future['year'])[::-1],
        y=list(future['Optimistic (Policy Reforms)'])+list(future['Pessimistic (No Action)'])[::-1],
        fill='toself', fillcolor='rgba(79,195,247,0.07)',
        line=dict(color='rgba(0,0,0,0)'), name='Uncertainty Band'))
    fig.add_hline(y=72.2, line_dash='dash', line_color='#4fc3f7',
                  annotation_text='Current 72.2 yrs')
    fig.update_layout(height=450, plot_bgcolor='#f8fbff', paper_bgcolor='white',
                      font=dict(family='Source Sans 3'), legend=dict(orientation='h', y=-0.15),
                      xaxis_title="Year", yaxis_title="Predicted Life Expectancy (Years)",
                      margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 2050 Projected Outcomes")
    for col,(sc, bg) in zip(st.columns(3),[
        ('Optimistic (Policy Reforms)','#1b4332'),
        ('Business as Usual','#1a2744'),
        ('Pessimistic (No Action)','#4a1212')
    ]):
        val = future[future['year']==2050][sc].values[0]
        diff = val - 72.2
        sign = "+" if diff>0 else ""
        col.markdown(f"""<div class="prediction-result" style="background:linear-gradient(135deg,{bg},{bg}99)">
            <div style="font-size:0.85em;color:#aaa;text-transform:uppercase">{sc.split('(')[0]}</div>
            <div class="prediction-number">{val:.1f}</div>
            <div style="color:#aaa">years by 2050</div>
            <div style="color:#4fc3f7;margin-top:8px;font-weight:700">{sign}{diff:.1f} yrs from today</div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════
# PAGE 5: SHAP
# ═══════════════════════════════
elif "SHAP" in page:
    st.markdown('<div class="section-header"><h2>🔍 XAI — SHAP Feature Importance</h2>'
                '<p>What drives life expectancy? Based on real WHO data.</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="insight-box"><b>What is SHAP?</b> SHapley Additive exPlanations assigns each feature '
                'a real-valued importance score — how many years of life expectancy it adds or removes. '
                'This makes our XGBoost model fully transparent.</div>', unsafe_allow_html=True)

    friendly = {
        'adult_mortality': 'Adult Mortality Rate', 'income_composition_of_resources': 'Human Development Index',
        'hiv_aids': 'HIV/AIDS Deaths', 'schooling': 'Years of Schooling',
        'bmi': 'BMI / Nutrition', 'alcohol': 'Alcohol Consumption',
        'thinness_1_19': 'Child Thinness (1-19 yrs)', 'thinness_5_9': 'Child Thinness (5-9 yrs)',
        'gdp': 'GDP per Capita', 'diphtheria': 'Diphtheria Immunisation',
        'total_expenditure': 'Health Expenditure %', 'polio': 'Polio Immunisation',
        'percentage_expenditure': 'Health % of GDP', 'status_encoded': 'Developed/Developing',
        'hepatitis_b': 'Hepatitis B Immunisation', 'infant_deaths': 'Infant Deaths',
        'under_five_deaths': 'Under-5 Deaths', 'year': 'Year', 'population': 'Population'
    }
    shap_plot = shap_df.copy()
    shap_plot['label'] = shap_plot['feature'].map(lambda x: friendly.get(x, x.replace('_',' ').title()))
    shap_plot = shap_plot.sort_values('importance', ascending=True).tail(15)

    fig = go.Figure(go.Bar(x=shap_plot['importance'], y=shap_plot['label'],
                           orientation='h',
                           marker=dict(color=shap_plot['importance'], colorscale='Blues', showscale=False),
                           text=shap_plot['importance'].round(3), textposition='outside'))
    fig.update_layout(height=500, plot_bgcolor='#f8fbff', paper_bgcolor='white',
                      font=dict(family='Source Sans 3'),
                      xaxis_title="Mean |SHAP Value| (Years of Life Expectancy)",
                      margin=dict(l=10,r=80,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🏆 Top 3 Most Impactful Factors (Real WHO Data)")
    top3 = shap_df.head(3)
    for col,(icon,(_,row)) in zip(st.columns(3), zip(["🏆","🥈","🥉"], top3.iterrows())):
        fname = friendly.get(row['feature'], row['feature'].replace('_',' ').title())
        col.markdown(f'<div class="metric-card"><div style="font-size:2em">{icon}</div>'
                     f'<div style="font-size:0.95em;font-weight:700;color:#e0f0ff;margin:8px 0 4px">{fname}</div>'
                     f'<div style="font-size:1.4em;color:#4fc3f7;font-weight:700">{row["importance"]:.3f}</div>'
                     f'<div style="font-size:0.75em;color:#94b8d4">Avg SHAP impact (years)</div></div>', unsafe_allow_html=True)


# ═══════════════════════════════
# PAGE 6: CAUSAL MODELLING
# ═══════════════════════════════
elif "Causal" in page:
    st.markdown('<div class="section-header"><h2>🔗 Causal Modelling — DoWhy</h2>'
                '<p>Proving causation, not just correlation — using real WHO data</p></div>', unsafe_allow_html=True)

    st.markdown('<div class="insight-box"><b>Why Causal AI?</b> SHAP tells us which features matter. '
                'DoWhy tells us <em>by how many years</em> a change in one factor <em>causes</em> '
                'life expectancy to change — critical for policy decisions.</div>', unsafe_allow_html=True)

    eff_df = pd.DataFrame(causal_edges)
    eff_df['color_bar'] = eff_df['effect'].apply(lambda x: '#81c784' if x>0 else '#e57373')
    eff_df['label'] = eff_df['from']

    fig = go.Figure(go.Bar(x=eff_df['label'], y=eff_df['effect'],
                           marker_color=eff_df['color_bar'],
                           text=eff_df['effect'].apply(lambda x: f"+{x:.1f}" if x>0 else f"{x:.1f}"),
                           textposition='outside'))
    fig.add_hline(y=0, line_color='#333', line_width=1)
    fig.update_layout(height=360, plot_bgcolor='#f8fbff', paper_bgcolor='white',
                      font=dict(family='Source Sans 3'), xaxis_tickangle=-20,
                      yaxis_title="Causal Effect on Life Expectancy (Years)",
                      margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Key Causal Findings")
    c1,c2 = st.columns(2)
    for col, edges in zip([c1,c2], [eff_df[eff_df['effect']>0], eff_df[eff_df['effect']<0]]):
        heading = "✅ Positive Causal Effects" if edges['effect'].mean()>0 else "❌ Negative Causal Effects"
        col.markdown(f"**{heading}**")
        for _,row in edges.iterrows():
            sign = "+" if row['effect']>0 else ""
            col.markdown(f"- **{row['from']}** → {sign}{row['effect']} years")


# ═══════════════════════════════
# PAGE 7: PERSONAL PREDICTOR
# ═══════════════════════════════
elif "Personal" in page:
    st.markdown('<div class="section-header"><h2>🧮 Personal Life Expectancy Predictor</h2>'
                '<p>Uses the real WHO-trained XGBoost model — enter your country\'s/region\'s indicators</p></div>',
                unsafe_allow_html=True)

    st.markdown('<div class="insight-box">⚠️ <b>Disclaimer:</b> Research prototype only. '
                'Not a medical diagnostic tool. Predictions based on population-level WHO data.</div>',
                unsafe_allow_html=True)

    model, rf, features, medians = load_model()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 📊 Health & Mortality")
        adult_mortality = st.slider("Adult Mortality (per 1000 adults)", 1, 500, 180)
        infant_deaths   = st.slider("Infant Deaths (per 1000 births)", 0, 200, 30)
        u5_deaths       = st.slider("Under-5 Deaths (per 1000)", 0, 300, 40)
        hiv_aids        = st.slider("HIV/AIDS Deaths (per 1000)", 0.01, 50.0, 0.1, 0.01)

    with c2:
        st.markdown("#### 🏥 Healthcare & Economy")
        schooling       = st.slider("Years of Schooling", 0.0, 20.0, 12.0, 0.5)
        hdi             = st.slider("Human Development Index (0–1)", 0.0, 1.0, 0.65, 0.01)
        health_exp      = st.slider("Govt Health Expenditure (%)", 1.0, 15.0, 5.0, 0.5)
        gdp             = st.slider("GDP per Capita (USD)", 100, 80000, 2000, 100)
        status          = st.selectbox("Country Status", ["Developing", "Developed"])

    with c3:
        st.markdown("#### 💉 Immunisation & Lifestyle")
        diphtheria      = st.slider("Diphtheria Immunisation (%)", 2, 100, 85)
        polio           = st.slider("Polio Immunisation (%)", 3, 100, 85)
        hepb            = st.slider("Hepatitis B Immunisation (%)", 2, 100, 80)
        bmi             = st.slider("Average BMI", 10.0, 45.0, 23.0, 0.5)
        alcohol         = st.slider("Alcohol Consumption (litres/year)", 0.0, 15.0, 3.0, 0.1)
        year_val        = st.slider("Year", 2000, 2030, 2024)

    if st.button("🔮 Predict Life Expectancy", type="primary"):
        row = {f: float(medians.get(f, 0)) for f in features}
        row.update({
            'adult_mortality': adult_mortality, 'infant_deaths': infant_deaths,
            'under_five_deaths': u5_deaths, 'hiv_aids': hiv_aids,
            'schooling': schooling, 'income_composition_of_resources': hdi,
            'total_expenditure': health_exp, 'gdp': gdp,
            'status_encoded': 1 if status=="Developed" else 0,
            'diphtheria': diphtheria, 'polio': polio, 'hepatitis_b': hepb,
            'bmi': bmi, 'alcohol': alcohol, 'year': year_val,
            'percentage_expenditure': (health_exp/100)*gdp
        })
        X_pred = pd.DataFrame([row])[features]
        xgb_pred = float(model.predict(X_pred)[0])
        rf_pred  = float(rf.predict(X_pred)[0])
        avg_pred = (xgb_pred + rf_pred) / 2

        c1p, c2p, c3p = st.columns(3)
        for col, (label, val, clr) in zip([c1p,c2p,c3p],[
            ("XGBoost Prediction", xgb_pred, "#4fc3f7"),
            ("Random Forest", rf_pred, "#81c784"),
            ("Ensemble Average", avg_pred, "#ffb74d")
        ]):
            col.markdown(f"""<div class="prediction-result">
                <div style="font-size:0.85em;color:#aaa;text-transform:uppercase">{label}</div>
                <div class="prediction-number" style="color:{clr}">{val:.1f}</div>
                <div style="color:#aaa">years</div>
                <div style="color:{clr};margin-top:8px;font-weight:700">
                    {"🟢 Above" if val>72.2 else "🔴 Below"} India avg (72.2 yrs)
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("### 💡 Key Recommendations to Improve")
        recs = []
        if adult_mortality > 200: recs.append("📉 **Reduce adult mortality** — improve emergency care and chronic disease management")
        if hiv_aids > 1.0:        recs.append("💊 **Address HIV/AIDS** — expand antiretroviral access")
        if schooling < 10:        recs.append("📚 **Increase education** — each extra year of schooling adds ~0.4 years of life")
        if hdi < 0.6:             recs.append("💰 **Improve HDI** — income + education + healthcare access together")
        if health_exp < 5:        recs.append("🏥 **Increase health expenditure** — currently below WHO recommended 5%")
        if diphtheria < 80:       recs.append("💉 **Boost immunisation rates** — target 90%+ for diphtheria and polio")
        if alcohol > 8:           recs.append("🚫 **Reduce alcohol consumption** — strongly linked to reduced lifespan")
        if not recs:              recs.append("✅ All indicators look strong! Focus on maintaining these levels.")
        for r in recs:
            st.markdown(f"- {r}")


# ═══════════════════════════════
# PAGE 8: MODEL COMPARISON
# ═══════════════════════════════
elif "Model Comparison" in page:
    st.markdown('<div class="section-header"><h2>📊 Model Comparison</h2>'
                '<p>XGBoost vs Random Forest — on real WHO data</p></div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    for col,(lbl,val) in zip([c1,c2,c3,c4],[
        ("XGBoost R²",     f"{metrics['xgb_r2']}"),
        ("XGBoost Acc",    f"{metrics['xgb_accuracy_pct']}%"),
        ("Random Forest R²",f"{metrics['rf_r2']}"),
        ("RF Accuracy",    f"{metrics['rf_accuracy_pct']}%"),
    ]):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div>'
                     f'<div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Actual vs Predicted scatter
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**XGBoost — Actual vs Predicted**")
        fig = px.scatter(avp, x='actual', y='xgb_predicted',
                         color_discrete_sequence=['#4fc3f7'], opacity=0.6)
        fig.add_shape(type='line', x0=avp['actual'].min(), y0=avp['actual'].min(),
                      x1=avp['actual'].max(), y1=avp['actual'].max(),
                      line=dict(color='#e57373', dash='dash'))
        fig.update_layout(plot_bgcolor='#f8fbff', paper_bgcolor='white',
                          xaxis_title="Actual", yaxis_title="Predicted",
                          font=dict(family='Source Sans 3'), margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Random Forest — Actual vs Predicted**")
        fig2 = px.scatter(avp, x='actual', y='rf_predicted',
                          color_discrete_sequence=['#81c784'], opacity=0.6)
        fig2.add_shape(type='line', x0=avp['actual'].min(), y0=avp['actual'].min(),
                       x1=avp['actual'].max(), y1=avp['actual'].max(),
                       line=dict(color='#e57373', dash='dash'))
        fig2.update_layout(plot_bgcolor='#f8fbff', paper_bgcolor='white',
                           xaxis_title="Actual", yaxis_title="Predicted",
                           font=dict(family='Source Sans 3'), margin=dict(l=10,r=10,t=20,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    # Summary table
    st.markdown("### Summary")
    summary = pd.DataFrame({
        'Model': ['XGBoost', 'Random Forest'],
        'R² Score': [metrics['xgb_r2'], metrics['rf_r2']],
        'Accuracy (%)': [metrics['xgb_accuracy_pct'], metrics['rf_accuracy_pct']],
        'MAE (years)': [metrics['xgb_mae'], metrics['rf_mae']],
        'Best For': ['Complex interactions, XAI via SHAP', 'Stability, ensemble averaging']
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)


# ═══════════════════════════════
# PAGE 9: ANCESTRAL INSIGHTS
# ═══════════════════════════════
elif "Ancestral" in page:
    st.markdown('<div class="section-header"><h2>📜 Ancestral Wisdom vs Modern Lifestyle</h2>'
                '<p>What our ancestors did right — backed by modern science</p></div>', unsafe_allow_html=True)

    for era, icon, practices, impact in [
        ("Vedic Period (~1500 BCE)", "🌿",
         ["Ayurvedic medicine — personalised treatment","Seasonal eating (Ritucharya)",
          "Daily yoga and pranayama","Plant-based diet, no processed food"],
         "Documented lifespans of 80–100 years among healthy adults"),
        ("Ancient India (Classical)", "☀️",
         ["Sun exposure — natural Vitamin D","Sleep with sunset, rise at sunrise",
          "Fasting practices (Ekadashi)","Herbal remedies: turmeric, neem, ashwagandha"],
         "Lower inflammation, better metabolism, stronger immunity"),
        ("Traditional Rural India", "🌾",
         ["Physical farming — 6+ hrs daily activity","Fresh, unprocessed local foods",
          "Strong community bonds (lower stress)","No alcohol or tobacco use"],
         "Rural India still has lower obesity and cardiac disease rates"),
        ("Modern Evidence-Based", "🔬",
         ["Mediterranean-Indian fusion diet","Intermittent fasting (science-backed)",
          "Mindfulness — proven stress reduction","Early disease screening (AI-powered)"],
         "Projected +5–8 year life extension if widely adopted"),
    ]:
        with st.expander(f"{icon}  {era}", expanded=False):
            c1,c2 = st.columns([2,1])
            with c1:
                for p in practices: st.markdown(f"- {p}")
            with c2:
                st.markdown(f'<div class="insight-box"><b>📈 Impact</b><br>{impact}</div>', unsafe_allow_html=True)

    st.markdown("### 🆚 Then vs Now — Radar Chart")
    comparison = pd.DataFrame({
        'Factor':['Physical Activity','Diet Quality','Stress Level','Sleep Quality',
                  'Community Bonds','Pollution','Processed Food','Healthcare'],
        'Ancestors':[9,9,3,9,9,1,1,5],
        'Modern India':[3,5,8,5,5,7,8,6]
    })
    fig = go.Figure()
    cats = comparison['Factor'].tolist()
    fig.add_trace(go.Scatterpolar(r=comparison['Ancestors'], theta=cats, fill='toself',
                                  name='Ancestors', line_color='#81c784',
                                  fillcolor='rgba(129,199,132,0.2)'))
    fig.add_trace(go.Scatterpolar(r=comparison['Modern India'], theta=cats, fill='toself',
                                  name='Modern India', line_color='#e57373',
                                  fillcolor='rgba(229,115,115,0.2)'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,10])),
                      height=420, paper_bgcolor='white', font=dict(family='Source Sans 3'),
                      legend=dict(orientation='h', y=-0.1))
    st.plotly_chart(fig, use_container_width=True)
