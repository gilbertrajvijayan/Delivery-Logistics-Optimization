# 🚚 Delivery Logistics Optimization
### An End-to-End Framework for Delay Prediction, Cost Forecasting, and Partner Optimization

> **Author:** Gilbert Raj Vijayan

---

## 📌 About the Project
This project builds a complete, end-to-end machine learning pipeline for 
delivery logistics using **PySpark** and **MLlib**. It tackles real-world 
logistics challenges — predicting delivery delays, forecasting costs and 
delivery times, segmenting shipments, and providing intelligent 
recommendations through an **LLM-powered delivery assistant**.

The project is built on a realistic synthetic dataset of **25,000 last-mile 
delivery records** from India, covering diverse conditions such as weather, 
vehicle types, distances, and partner performance.

---

## 🎯 Problem Statement
Modern logistics companies face:
- Unpredictable delivery delays
- Inaccurate cost and time estimates
- Poor delivery partner selection
- Reactive rather than proactive decision-making

This project solves all four using machine learning and LLM-based automation.

---

## 📂 Files
| File | Description |
|------|-------------|
| `Delivery_Logistics.ipynb` | Full PySpark ML pipeline (Google Colab) |
| `Delivery_Logistics_report.docx` | Detailed project report |
| `Delivery_Logistics_PPT.pptx` | Project presentation slides |

---

## 🧠 Models Built

### Model 1 — Delay Prediction (Classification)
- Algorithms: **Random Forest, Logistic Regression, Gradient Boosted Trees**
- AUC: **0.953 – 0.958** | Accuracy: **~95%**
- Key features: delivery rating, expected time, express mode, vehicle type

### Model 2 — Delivery Time Prediction (Regression)
- Algorithm: **Linear Regression**
- Result: Near-perfect R² with very low RMSE
- Key features: distance, weight, cost per km

### Model 3 — Delivery Cost Prediction (Regression)
- Algorithm: **Linear Regression**
- Result: Near-perfect R² with very low RMSE
- Enables accurate cost planning and customer quotes

### Model 4 — Shipment Segmentation (Clustering)
- Algorithm: **K-Means (3 clusters)**
- Clusters identified:
  - 🟦 Light-weight long-distance shipments
  - 🟩 Heavy-weight long-distance shipments
  - 🟨 Short-distance mid-weight shipments (lowest delay rate)

### Model 5 — LLM-Powered Delivery Assistant 🤖
- Answers real-time logistics queries:
  - *"Will this delivery be delayed?"*
  - *"How long will this shipment take?"*
  - *"Which delivery partner should I choose?"*
- Example output: On-time prediction, delay probability: 0.12, 
  estimated time: 5.1 hrs, recommended partner: Best performer

---

## 📊 Dataset
- **Size:** 25,000 delivery records
- **Source:** Synthetic realistic dataset (India last-mile delivery)
- **Key Features:**
  - `distance_km` — Distance covered (1–300 km)
  - `package_weight_kg` — Package weight (0.2–50 kg)
  - `expected_time_hrs` — Expected delivery time
  - `weather_condition` — Stormy, rainy, clear, etc.
  - `vehicle_type` — Bike, van, truck, etc.
  - `delivery_partner` — Partner assigned
- **Target Variables:**
  - `delay_label_noisy` — Binary delay flag
  - `delivery_time_hrs_noisy` — Actual delivery time (hrs)
  - `delivery_cost_noisy` — Actual delivery cost

---

## 🔑 Key Findings
- ⛈️ Stormy and rainy weather significantly increases delivery times
- 📍 Long-distance shipments naturally have higher delivery times
- 💰 Same-day deliveries are the most costly
- ⭐ Delivery ratings are right-skewed — most deliveries are rated highly
- 🏆 Stacking Classifier outperformed all individual models

---

## 🛠️ Tech Stack
| Tool | Purpose |
|------|---------|
| PySpark + MLlib | Big data ML pipelines |
| Python (Pandas, NumPy) | Data manipulation |
| Google Colab | Development environment |
| Matplotlib / Seaborn | Visualizations |
| LLM Integration | Intelligent delivery assistant |

---

## 💡 Business Impact
- Reduce delivery delays proactively
- Accurate cost and time estimates for customers
- Smarter delivery partner selection
- Bridges gap between technical models and business decisions
