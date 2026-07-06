# 🏠 Dubai/Abu Dhabi Real Estate Price Intelligence Tool

An end-to-end machine learning tool that estimates fair market prices for residential properties in Dubai and Abu Dhabi, and explains WHY using explainable AI.

## Problem

Retail investors and first-time buyers struggle to judge fair pricing in the UAE real estate market due to fragmented data across sources like DLD, Bayut, and Property Finder. This tool consolidates public transaction data into a single, transparent pricing model.

## Features

- **Price Prediction** — Input property details (area, type, size, rooms, parking) to get an estimated fair price
- **Verdict System** — Compares an asking price against the model's estimate to flag listings as overpriced, underpriced, or fair
- **Explainability** — SHAP-based breakdown showing exactly which features (location, size, rooms) are driving the predicted price up or down

## Tech Stack

- **Data**: Dubai Land Department (DLD) public transaction records (115K+ raw rows, cleaned to ~73K)
- **Model**: CatBoost Regressor
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Frontend**: Streamlit
- **Deployment**: Streamlit Community Cloud

## Model Performance

- **R²**: 0.849
- **MAE**: ~326,000 AED (~8.8% error)
- **RMSE**: ~677,000 AED

## Key Insights (via SHAP)

The strongest drivers of price, in order:

1. **Area/Location** (`area_en`) — by far the biggest factor
2. **Actual Size** (`actual_area`)
3. **Number of Rooms**
4. **Property Sub-Type**

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Background

Built as a portfolio project bridging Business Analyst, Product Manager, AI Engineer, and Cloud skill sets — covering the full pipeline from data cleaning and feature engineering to model training, explainability, and deployment.

## Author

Jefia Edwin — www.linkedin.com/in/jef4450 — https://www.notion.so/JOB-TRACKER-39319b39450f80569c7cc8ec178df878?source=copy_link
