"""
Streamlit dashboard — Economies Recovery After Shocks
=====================================================
Tabs:
  1. 🗺️  Global Map      – choropleth for any metric × year
  2. 🔍  Country Deep Dive – GDP timeline + model predictions + what-if predictor
  3. 📊  Model Comparison  – metrics table, bar charts, feature importance
  4. 📈  Recovery Analysis – indexed GDP trajectories + recovery rankings
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import (
    load_dashboard,
    load_metrics,
    load_predictions,
    load_xgb,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Economies Recovery Dashboard",
    page_icon="🌍",
    layout="wide",
)

# ── Colour palette — Ensemble removed ────────────────────────────────────────
MODEL_COLORS = {
    "Naive":    "#aaaaaa",
    "HistMean": "#cccccc",
    "XGBoost":  "#2196F3",
    "ARIMA":    "#FF9800",
    "LSTM":     "#4CAF50",
}

# ── Load data (cached) ────────────────────────────────────────────────────────
dashboard    = load_dashboard()
metrics_full = load_metrics()
preds        = load_predictions()   # should no longer contain "Ensemble" key
xgb_model, le = load_xgb()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌍 Economies Recovery After Shocks")
st.caption(
    "GDP per capita forecasting · 212 countries · "
    "Individual model test period: 2019–2024 · Final comparison: aligned 2020–2024 · "
    "Models: XGBoost, ARIMA, LSTM"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["🗺️ Global Map", "🔍 Country Deep Dive", "📊 Model Comparison", "📈 Recovery Analysis"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — GLOBAL MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Global Economic Indicators")

    ctrl1, ctrl2 = st.columns([3, 1])
    with ctrl1:
        metric_options = {
            "GDP per Capita (USD)": "gdp_per_capita",
            "GDP Growth Rate (%)":  "gdp_growth_rate",
            "Inflation — HCPI (%)": "inflation_hcpi",
            "Employment Rate (%)":  "employment_rate",
            "Dependency Ratio":     "dependency_ratio",
            "Country Cluster":      "cluster_name",
        }
        metric_label = st.selectbox("Metric", list(metric_options.keys()), key="map_metric")
        metric_col   = metric_options[metric_label]
    with ctrl2:
        map_year = st.slider("Year", 1990, 2024, 2023, key="map_year")

    year_data = dashboard[dashboard["year"] == map_year].copy()

    # ── Choropleth ────────────────────────────────────────────────────────────
    if metric_col == "cluster_name":
        fig_map = px.choropleth(
            year_data,
            locations="country_code",
            color="cluster_name",
            hover_name="country_code",
            hover_data={"gdp_per_capita": ":,.0f"},
            color_discrete_sequence=px.colors.qualitative.Set2,
            title=f"Country Clusters — {map_year}",
            labels={"cluster_name": "Cluster"},
        )
    else:
        reverse = metric_col in ("inflation_hcpi", "dependency_ratio")
        fig_map = px.choropleth(
            year_data,
            locations="country_code",
            color=metric_col,
            hover_name="country_code",
            hover_data={metric_col: ":.2f"},
            color_continuous_scale="RdYlGn_r" if reverse else "RdYlGn",
            title=f"{metric_label} — {map_year}",
            labels={metric_col: metric_label},
        )

    fig_map.update_layout(
        geo=dict(showframe=False, showcoastlines=True, projection_type="equirectangular"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=480,
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ── Stats row — FIX: handle categorical (cluster_name) separately ─────────
    if metric_col == "cluster_name":
        # Categorical column — show counts per cluster, not numeric stats
        valid_cat = year_data["cluster_name"].dropna()
        if len(valid_cat):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Countries with data", f"{len(valid_cat):,}")
            value_counts = valid_cat.value_counts()
            c2.metric("Unique clusters",     f"{valid_cat.nunique()}")
            c3.metric("Largest cluster",     f"{value_counts.index[0]}")
            c4.metric("Cluster size",        f"{value_counts.iloc[0]:,} countries")
    else:
        # Numeric column — show mean / median / range as before
        valid_num = year_data[metric_col].dropna()
        if len(valid_num):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Countries with data", f"{len(valid_num):,}")
            c2.metric("Mean",   f"{valid_num.mean():,.2f}")
            c3.metric("Median", f"{valid_num.median():,.2f}")
            c4.metric("Range",  f"{valid_num.min():,.2f} – {valid_num.max():,.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — COUNTRY DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    countries  = sorted(dashboard["country_code"].unique())
    default_ix = countries.index("USA") if "USA" in countries else 0
    country    = st.selectbox("Select Country", countries, index=default_ix, key="country_sel")

    c_hist = dashboard[dashboard["country_code"] == country].sort_values("year")

    # ── GDP timeline ──────────────────────────────────────────────────────────
    st.subheader(f"GDP per Capita — {country}")

    fig_gdp = go.Figure()

    # Shade shock years
    for yr in c_hist[c_hist["is_shock"] == 1]["year"].tolist():
        fig_gdp.add_vrect(
            x0=yr - 0.45, x1=yr + 0.45,
            fillcolor="red", opacity=0.10, layer="below", line_width=0,
        )

    # Model predictions — only models that exist in preds dict
    for mname, pred_df in preds.items():
        if mname not in MODEL_COLORS:
            continue   # skip any leftover Ensemble key gracefully
        cp = pred_df[pred_df["country_code"] == country].sort_values("pred_year")
        if cp.empty:
            continue
        fig_gdp.add_trace(go.Scatter(
            x=cp["pred_year"], y=cp["predicted"],
            mode="lines+markers", name=f"{mname} (pred)",
            line=dict(
                color=MODEL_COLORS[mname], width=1.5,
                dash="dash" if mname in ("Naive", "HistMean") else "solid",
            ),
            marker=dict(size=5, symbol="diamond"),
        ))

    # Actual historical GDP (black, on top)
    fig_gdp.add_trace(go.Scatter(
        x=c_hist["year"], y=c_hist["gdp_per_capita"],
        mode="lines+markers", name="Actual",
        line=dict(color="black", width=2.5),
        marker=dict(size=4),
    ))

    fig_gdp.add_vline(
        x=2018.5, line_dash="dot", line_color="gray",
        annotation_text="Train | Test", annotation_position="top right",
    )
    fig_gdp.update_layout(
        xaxis_title="Year", yaxis_title="GDP per Capita (USD)",
        legend=dict(orientation="h", y=-0.28, font_size=11),
        height=420, margin=dict(l=0, r=0, t=10, b=0),
        hovermode="x unified",
    )
    st.plotly_chart(fig_gdp, use_container_width=True)

    # Quick stats row
    latest = c_hist.iloc[-1]
    first  = c_hist.iloc[0]
    pct_ch = (latest["gdp_per_capita"] - first["gdp_per_capita"]) / first["gdp_per_capita"] * 100
    s1, s2, s3, s4 = st.columns(4)
    s1.metric(f"Latest GDP/cap ({int(latest['year'])})", f"${latest['gdp_per_capita']:,.0f}")
    s2.metric("Growth since 1990", f"{pct_ch:+.1f}%")
    s3.metric("Avg GDP/cap (1990–2024)", f"${c_hist['gdp_per_capita'].mean():,.0f}")
    cluster_val = latest.get("cluster_name", "N/A")
    s4.metric("K-Means Cluster", str(cluster_val))

    st.divider()

    # ── What-If Predictor ─────────────────────────────────────────────────────
    st.subheader("🤖 What-If GDP Predictor (XGBoost)")
    st.caption(
        "Choose an anchor year and adjust economic conditions to predict the following year's GDP. "
        "Sliders initialise to actual recorded values."
    )

    anchor_range = [y for y in sorted(c_hist["year"].tolist()) if 2018 <= y <= 2023]

    if not anchor_range or country not in le.classes_:
        st.warning("What-if predictor unavailable — country not in XGBoost training set.")
    else:
        wi_col_a, wi_col_b = st.columns([3, 1])
        with wi_col_a:
            anchor_year = st.select_slider(
                "Anchor year (predicts the following year's GDP)",
                options=anchor_range, value=anchor_range[-1], key="wi_anchor",
            )
        with wi_col_b:
            if st.button("↺ Reset sliders", key="wi_reset"):
                for k in ["wi_inf", "wi_emp", "wi_dep", "wi_wg", "wi_pct"]:
                    st.session_state.pop(k, None)
                st.rerun()

        row = c_hist[c_hist["year"] == anchor_year]
        if row.empty:
            st.warning("No data for selected anchor year.")
        else:
            row = row.iloc[0]

            def _fval(col, fallback):
                v = row.get(col)
                return float(v) if pd.notna(v) else fallback

            w1, w2, w3 = st.columns(3)
            with w1:
                inflation  = st.slider("Inflation — HCPI (%)", -5.0, 50.0, _fval("inflation_hcpi", 2.0),  0.1, key="wi_inf")
                employment = st.slider("Employment Rate (%)",   30.0, 95.0, _fval("employment_rate", 60.0), 0.1, key="wi_emp")
            with w2:
                dependency   = st.slider("Dependency Ratio",     15.0, 120.0, _fval("dependency_ratio", 50.0), 0.5, key="wi_dep")
                is_shock_wi  = st.radio(
                    "Is Shock Year?", [0, 1],
                    index=int(_fval("is_shock", 0)),
                    horizontal=True, key="wi_shock",
                    format_func=lambda x: "No" if x == 0 else "Yes ⚡",
                )
            with w3:
                world_growth = st.slider("World GDP Growth (%)",      -8.0,  8.0,  _fval("world_gdp_growth", 2.5),           0.1, key="wi_wg")
                pct_contract = st.slider("% Countries Contracting",    0.0, 100.0, _fval("pct_countries_contracting", 20.0), 0.5, key="wi_pct")

            gdp_now  = float(row["gdp_per_capita"])

            def _lookup_gdp(yr):
                rows = c_hist[c_hist["year"] == yr]["gdp_per_capita"].values
                return float(rows[0]) if len(rows) else None

            gdp_lag1 = _lookup_gdp(anchor_year - 1) or gdp_now
            gdp_lag2 = _lookup_gdp(anchor_year - 2) or gdp_lag1
            gdp_gr   = (gdp_now - gdp_lag1) / gdp_lag1 if gdp_lag1 > 0 else 0.0
            gr_lag1  = (gdp_lag1 - gdp_lag2) / gdp_lag2 if gdp_lag2 > 0 else 0.0
            roll_mn  = (gdp_now + gdp_lag1 + gdp_lag2) / 3.0
            c_enc    = int(le.transform([country])[0])

            feat = pd.DataFrame([{
                "country_encoded":           c_enc,
                "year":                      anchor_year,
                "gdp_per_capita":            gdp_now,
                "gdp_lag_1":                 gdp_lag1,
                "gdp_lag_2":                 gdp_lag2,
                "gdp_growth_rate":           gdp_gr,
                "growth_lag_1":              gr_lag1,
                "gdp_rolling_mean_3yr":      roll_mn,
                "inflation_hcpi":            inflation,
                "employment_rate":           employment,
                "dependency_ratio":          dependency,
                "is_shock":                  is_shock_wi,
                "world_gdp_growth":          world_growth,
                "pct_countries_contracting": pct_contract,
            }])

            pred_gdp = float(xgb_model.predict(feat)[0])
            delta    = pred_gdp - gdp_now

            xgb_next_rows   = preds.get("XGBoost", pd.DataFrame())
            actual_next_row = (
                xgb_next_rows[
                    (xgb_next_rows["country_code"] == country) &
                    (xgb_next_rows["pred_year"] == anchor_year + 1)
                ]
                if not xgb_next_rows.empty else pd.DataFrame()
            )

            r1, r2, r3 = st.columns(3)
            r1.metric(f"Current GDP/cap ({anchor_year})", f"${gdp_now:,.0f}")
            r2.metric(
                f"Predicted GDP/cap ({anchor_year + 1})",
                f"${pred_gdp:,.0f}",
                f"{delta:+,.0f}  ({delta / gdp_now * 100:+.1f}%)",
            )
            if not actual_next_row.empty:
                actual_next = float(actual_next_row["actual"].values[0])
                err = pred_gdp - actual_next
                r3.metric(
                    f"Actual GDP/cap ({anchor_year + 1})",
                    f"${actual_next:,.0f}",
                    f"Model error: {err:+,.0f}",
                    delta_color="off",
                )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL COMPARISON  (Ensemble removed)
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model Performance Metrics")
    st.caption("Showing: **Aligned Test Set (2020–2024)** · Common country-year window across all models")

    def highlight_best(df):
        styles = pd.DataFrame("", index=df.index, columns=df.columns)
        for col in df.columns:
            if col in ("MAE", "RMSE", "MAPE", "MASE"):
                styles.loc[df[col].idxmin(), col] = "background-color:#c8f7c5;font-weight:bold"
            elif col == "DA":
                styles.loc[df[col].idxmax(), col] = "background-color:#c8f7c5;font-weight:bold"
        return styles

    # ── Drop Ensemble row if it somehow still exists in the metrics CSV ───────
    mdf = metrics_full.drop(
        index=[i for i in metrics_full.index if "ensemble" in str(i).lower()],
        errors="ignore",
    )

    st.dataframe(
        mdf.style.apply(highlight_best, axis=None).format("{:.4f}"),
        use_container_width=True,
    )

    st.divider()

    model_order = mdf.index.tolist()
    bc1, bc2 = st.columns(2)

    with bc1:
        fig_mae = px.bar(
            mdf.reset_index().rename(columns={"index": "Model"}),
            x="Model", y="MAE",
            color="Model", color_discrete_map=MODEL_COLORS,
            text="MAE",
            title="Mean Absolute Error — Aligned Test Set (2020–2024)",
            category_orders={"Model": model_order},
        )
        fig_mae.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
        fig_mae.update_layout(
            showlegend=False, yaxis_title="MAE (USD)",
            height=380, margin=dict(t=50, b=0),
        )
        st.plotly_chart(fig_mae, use_container_width=True)

    with bc2:
        fig_da = px.bar(
            mdf.reset_index().rename(columns={"index": "Model"}),
            x="Model", y="DA",
            color="Model", color_discrete_map=MODEL_COLORS,
            text="DA",
            title="Directional Accuracy — Aligned Test Set (2020–2024)",
            category_orders={"Model": model_order},
        )
        fig_da.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_da.add_hline(
            y=50, line_dash="dot", line_color="gray",
            annotation_text="50% random baseline", annotation_position="right",
        )
        fig_da.update_layout(
            showlegend=False, yaxis_title="DA (%)", yaxis_range=[0, 110],
            height=380, margin=dict(t=50, b=0),
        )
        st.plotly_chart(fig_da, use_container_width=True)

    st.divider()

    # ── Classification metrics ─────────────────────────────────────────────────
    st.subheader("Classification Metrics — Direction Prediction (GDP Up/Down)")
    st.caption("Directional Accuracy (DA), confusion matrix, precision, recall and F1-score "
    "for GDP up/down prediction.")

    try:
        import os
        class_metrics_path = os.path.join("output", "phase7_classification_metrics.csv")
        if os.path.exists(class_metrics_path):
            class_df = pd.read_csv(class_metrics_path, index_col=0)

            # Drop Ensemble row if present
            class_df = class_df.drop(
                index=[i for i in class_df.index if "ensemble" in str(i).lower()],
                errors="ignore",
            )

            st.dataframe(
                class_df.style.format("{:.2f}").highlight_max(
                    subset=[
                        "Directional Accuracy",
                        "Sensitivity (TPR)",
                        "Specificity (TNR)",
                        "F1-Score",
                    ],
                    color="#c8f7c5",
                ),
                use_container_width=True,
            )

            img_path = os.path.join("output", "phase7_confusion_matrices.png")
            if os.path.exists(img_path):
                from PIL import Image
                img = Image.open(img_path)
                st.image(img, caption="Confusion Matrices (TP/TN/FP/FN)")
        else:
            st.info("ℹ️ Run Phase 7 notebook to generate classification metrics.")
    except Exception as e:
        st.warning(f"Could not load classification metrics: {e}")

    st.divider()

    # ── Feature importance ─────────────────────────────────────────────────────
    st.subheader("XGBoost Feature Importance")
    fi = pd.Series(
        xgb_model.feature_importances_, index=xgb_model.feature_names_in_
    ).sort_values()
    fig_fi = px.bar(
        x=fi.values, y=fi.index, orientation="h",
        color=fi.values, color_continuous_scale="Blues",
        title="Feature Importance — XGBoost",
        labels={"x": "Importance", "y": "Feature"},
        text=fi.round(4).values,
    )
    fig_fi.update_traces(texttemplate="%{text}", textposition="outside")
    fig_fi.update_layout(
        showlegend=False, coloraxis_showscale=False,
        height=430, margin=dict(l=0, r=80, t=50, b=0),
    )
    st.plotly_chart(fig_fi, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — RECOVERY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Economic Recovery Trajectories")

    shock_choice = st.radio(
        "Shock event",
        ["2008 Global Recession", "2020 COVID-19"],
        horizontal=True, key="shock_choice",
    )

    if shock_choice == "2008 Global Recession":
        shock_year  = 2008
        quality_col = "recovery_quality_recession_2008"
        speed_col   = "recovery_speed_recession_2008"
        chart_years = list(range(2005, 2020))
        speed_note  = "Years to recover (lower = faster)"
    else:
        shock_year  = 2020
        quality_col = "recovery_quality_covid_2020"
        speed_col   = "recovery_speed_covid_2020"
        chart_years = list(range(2018, 2025))
        speed_note  = "Recovery speed (1 = fastest, 4 = slowest / not recovered)"

    rec_summary = (
        dashboard.groupby("country_code")[[quality_col, speed_col]]
        .first()
        .dropna()
        .sort_values(quality_col, ascending=False)
        .reset_index()
        .rename(columns={
            "country_code": "Country",
            quality_col:    "Recovery Quality (%)",
            speed_col:      "Recovery Speed",
        })
    )
    avail_countries = rec_summary["Country"].tolist()
    default_sel     = sorted(avail_countries[:8])

    selected = st.multiselect(
        "Countries to plot (indexed to 100 at shock year)",
        sorted(avail_countries),
        default=default_sel,
        key="rec_countries",
    )

    if selected:
        fig_rec = go.Figure()

        for ctry in selected:
            c_data = dashboard[
                (dashboard["country_code"] == ctry) &
                (dashboard["year"].isin(chart_years))
            ].sort_values("year")

            base_rows = c_data[c_data["year"] == shock_year]["gdp_per_capita"].values
            if not len(base_rows) or pd.isna(base_rows[0]) or base_rows[0] == 0:
                continue

            base_gdp = base_rows[0]
            indexed  = c_data["gdp_per_capita"] / base_gdp * 100

            fig_rec.add_trace(go.Scatter(
                x=c_data["year"], y=indexed,
                mode="lines+markers", name=ctry,
                marker=dict(size=5),
                hovertemplate=f"<b>{ctry}</b>: %{{y:.1f}}<extra></extra>",
            ))

        fig_rec.add_hline(
            y=100, line_dash="dash", line_color="black",
            annotation_text=f"Shock year baseline ({shock_year} = 100)",
            annotation_position="right",
        )
        fig_rec.add_vline(
            x=shock_year, line_dash="dot", line_color="crimson", opacity=0.7,
            annotation_text="Shock", annotation_position="top right",
        )
        fig_rec.update_layout(
            title=f"Indexed GDP Recovery — {shock_choice}  ({shock_year} = 100)",
            xaxis_title="Year", yaxis_title="GDP Index (shock year = 100)",
            height=500, hovermode="x unified",
            legend=dict(orientation="h", y=-0.2),
        )
        st.plotly_chart(fig_rec, use_container_width=True)

    st.divider()

    t1, t2 = st.columns(2)
    fmt = {"Recovery Quality (%)": "{:.1f}", "Recovery Speed": "{:.0f}"}

    with t1:
        st.write(f"**🏆 Top 15 Best Recovery** — {shock_choice}")
        st.caption("Recovery Quality % = GDP at end of recovery window vs pre-shock baseline")
        st.dataframe(
            rec_summary.head(15).set_index("Country").style.format(fmt),
            use_container_width=True,
        )

    with t2:
        st.write(f"**⚠️ Bottom 15 Weakest Recovery** — {shock_choice}")
        st.caption(speed_note)
        st.dataframe(
            rec_summary.tail(15).iloc[::-1].set_index("Country").style.format(fmt),
            use_container_width=True,
        )