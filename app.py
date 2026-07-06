import streamlit as st
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import json
import shap
import matplotlib.pyplot as plt

# ── Page config ──────────────────────────────────────────
st.set_page_config(page_title="Dubai/AD Real Estate Price Intelligence", layout="wide")

# ── Load model and metadata (cached so it doesn't reload every interaction) ──
@st.cache_resource
def load_model():
    model = CatBoostRegressor()
    model.load_model('real_estate_price_model.cbm')
    return model

@st.cache_data
def load_feature_info():
    with open('feature_info.json', 'r') as f:
        return json.load(f)

@st.cache_data
def load_dropdown_values():
    with open('dropdown_values.json', 'r') as f:
        return json.load(f)

model = load_model()
feature_info = load_feature_info()
dropdown_values = load_dropdown_values()

# ── Title ─────────────────────────────────────────────────
st.title("🏠 Dubai / Abu Dhabi Real Estate Price Intelligence")
st.markdown("Get a fair price estimate for a property listing, with an explanation of what's driving the price.")

# ── Input form ────────────────────────────────────────────
st.header("Listing Details")

col1, col2 = st.columns(2)

with col1:
    area_en = st.selectbox("Area", options=dropdown_values['area_en'])
    prop_type_en = st.selectbox("Property Type", options=dropdown_values['prop_type_en'])
    prop_sb_type_en = st.selectbox("Property Sub-Type", options=dropdown_values['prop_sb_type_en'])
    actual_area = st.number_input("Actual Area (sq meters)", min_value=20, max_value=600, value=90, step=5)

with col2:
    rooms_num = st.number_input("Number of Rooms", min_value=0, max_value=15, value=2, step=1)
    has_parking = st.selectbox("Has Parking?", options=["Yes", "No"])
    year = st.number_input("Transaction Year", min_value=2000, max_value=2030, value=2026, step=1)
    month = st.number_input("Transaction Month", min_value=1, max_value=12, value=7, step=1)

st.divider()
asking_price = st.number_input("Asking/Listed Price (AED) — optional, for verdict comparison",
                                 min_value=0, value=0, step=10000)

# ── Predict button ────────────────────────────────────────
if st.button("Get Price Estimate", type="primary"):

    # Build input dataframe in the exact feature order the model expects
    input_data = pd.DataFrame([{
        'area_en': area_en,
        'prop_type_en': prop_type_en,
        'prop_sb_type_en': prop_sb_type_en,
        'actual_area': actual_area,
        'rooms_num': rooms_num,
        'has_parking': 1 if has_parking == "Yes" else 0,
        'year': year,
        'month': month
    }])

    # Ensure column order matches training
    input_data = input_data[feature_info['features']]

    # Predict
    predicted_price = model.predict(input_data)[0]

    # MAE from your model evaluation — used to build a "fair range"
    MAE = 326022  # update this if your latest model run gives a different number

    lower_bound = predicted_price - MAE
    upper_bound = predicted_price + MAE

    # ── Results ───────────────────────────────────────────
    st.header("Results")

    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Predicted Fair Price", f"AED {predicted_price:,.0f}")
    res_col2.metric("Fair Range (Low)", f"AED {lower_bound:,.0f}")
    res_col3.metric("Fair Range (High)", f"AED {upper_bound:,.0f}")

    # ── Verdict logic (only if asking price provided) ────
    if asking_price > 0:
        st.subheader("Verdict")
        diff_pct = ((asking_price - predicted_price) / predicted_price) * 100

        if asking_price < lower_bound:
            st.success(f"🟢 **UNDERPRICED** — listed {abs(diff_pct):.1f}% below fair value. Good deal for a buyer.")
        elif asking_price > upper_bound:
            st.error(f"🔴 **OVERPRICED** — listed {diff_pct:.1f}% above fair value.")
        else:
            st.info(f"🟡 **FAIR PRICE** — within {abs(diff_pct):.1f}% of estimated fair value.")

    # ── SHAP explanation ──────────────────────────────────
    st.subheader("What's Driving This Price?")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_data)[0]

    feature_names = input_data.columns.tolist()
    feature_values = input_data.iloc[0].values

    # Build labels like "area_en = AL BARARI"
    labels = [f"{feature_names[i]} = {feature_values[i]}" for i in range(len(feature_names))]

    # Sort by absolute impact, largest at top
    order = np.argsort(np.abs(shap_values))
    sorted_labels = [labels[i] for i in order]
    sorted_values = [shap_values[i] for i in order]
    colors = ['#ff4b4b' if v > 0 else '#3b82f6' for v in sorted_values]

    plt.close('all')

    # Big, tall figure with generous spacing between bars
    n = len(sorted_labels)
    fig, ax = plt.subplots(figsize=(14, max(6, n * 1.1)))

    # height < 1 leaves visible gaps between bars, less cramped
    bars = ax.barh(sorted_labels, sorted_values, color=colors, height=0.5)

    max_abs = max(np.abs(sorted_values))
    min_gap = max_abs * 0.25  # push every label well clear of the zero line

    for bar, val in zip(bars, sorted_values):
        x = bar.get_width()
        sign = 1 if val >= 0 else -1
        label_x = sign * max(abs(x), min_gap)
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+,.0f}",
            va='center',
            ha='left' if sign > 0 else 'right',
            fontsize=11,
            fontweight='bold',
            # white box behind text so it never visually merges with bars/gridlines
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5)
        )

    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel("Impact on Predicted Price (AED)", fontsize=12)
    ax.set_title(f"Base price: AED {explainer.expected_value:,.0f}", fontsize=13)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='x', labelsize=10)

    # Generous room on both sides for labels
    ax.set_xlim(
        min(0, min(sorted_values)) * 1.5 - min_gap,
        max(0, max(sorted_values)) * 1.5 + min_gap
    )

    plt.subplots_adjust(left=0.35, top=0.92, bottom=0.08)

    st.pyplot(fig)
    plt.close(fig)

    st.caption("Red = pushes price up, Blue = pushes price down. Bar length = impact strength.")