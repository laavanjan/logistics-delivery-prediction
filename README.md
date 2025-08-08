# 📈 Predictive Modelling of Delhivery Logistics

![cover](https://i.postimg.cc/mkKCL8Cf/cover-image.webp)

## 📅 Project Overview

This repository contains a Business Intelligence and Predictive Analytics project focused on enhancing the **logistics efficiency** of **Delhivery**, a leading supply chain company in India. The goal is to analyze delivery data, clean and preprocess it, and build an interactive dashboard and predictive model that provides insights into:

* Trip efficiency
* Route optimization
* Transportation types
* Overall delivery performance

---

## 📊 Objectives

1. **Data Cleaning and Preprocessing**: Handle missing values, convert types, and extract time-based features.
2. **Trip Efficiency Analysis**: Study patterns in trip durations and delays.
3. **Route Optimization Insights**: Analyze delivery routes to recommend better alternatives.
4. **Delivery Performance Metrics**: Track key performance indicators (KPIs).
5. **Forecasting Support**: Enable predictive models for delivery time estimation.

---

## 📚 Problem Statement

Improving delivery route planning and optimizing delivery schedules while ensuring timely and reliable service.

Build predictive models to estimate delivery times for different routes and time slots. Accurate delivery time estimation enhances customer satisfaction and enables Delhivery to provide reliable service commitments.

---

## 📊 Dataset Description

The dataset comprises **14817 unique trips** and **144867 total rows**. It includes:

* **24 attributes**:

  * **12 object/categorical columns**
  * **11 numerical columns**
  * **1 boolean column**
* **Target column**: `actual_time` (in minutes)
* **Format**: CSV

### 🔄 Features Table

| Feature                                      | Description                                   |
| -------------------------------------------- | --------------------------------------------- |
| `data`                                       | Indicates training/testing data               |
| `trip_creation_time`                         | Timestamp of trip creation                    |
| `route_schedule_uuid`                        | Unique ID for route schedule                  |
| `route_type`                                 | Type of transport (FTL/Carting)               |
| `trip_uuid`                                  | Unique trip identifier                        |
| `source_center`, `destination_center`        | Location IDs                                  |
| `source_name`, `destination_name`            | Location names                                |
| `od_start_time`, `od_end_time`               | Trip start and end times                      |
| `start_scan_to_end_scan`                     | Time between scans at source and destination  |
| `is_cutoff`                                  | Boolean field, needs investigation            |
| `cutoff_factor`, `cutoff_timestamp`          | Unknown fields, potential feature engineering |
| `actual_distance_to_destination`             | Distance (km) between centers                 |
| `actual_time`                                | Target column: time taken to deliver          |
| `osrm_time`, `osrm_distance`                 | Routing engine estimates for time/distance    |
| `factor`, `segment_factor`                   | Unknown; requires analysis                    |
| `segment_actual_time`                        | Time for a delivery segment                   |
| `segment_osrm_time`, `segment_osrm_distance` | OSRM estimates for segment                    |

---

## 🧰 ML and BI Strategy

### 🧮 Machine Learning

| Aspect                   | Description                                         |
| ------------------------ | --------------------------------------------------- |
| **Task**                 | Regression                                          |
| **Target**               | `actual_time`                                       |
| **Algorithms**           | Linear Regression, Random Forest, XGBoost, CatBoost |
| **Metrics**              | RMSE, MAE, R²                                       |
| **Feature Engineering**  | Delay %, Trip Efficiency, Time Slots                |
| **Categorical Encoding** | Required for route\_type, source/destination        |
| **Time Features**        | Peak hour, weekday trends, delivery duration        |

### 📊 Business Intelligence KPIs

| KPI               | Description                         |
| ----------------- | ----------------------------------- |
| Avg Delivery Time | Mean delivery duration              |
| Trip Efficiency   | Ratio of OSRM vs Actual time        |
| Delay Patterns    | By route, time of day, location     |
| Mode Performance  | FTL vs Carting metrics              |
| Route Volume      | Number of trips per route           |
| Delay Trends      | Time-based or location-based delays |

---

## 🔧 Tools and Technologies

* **Programming**: Python, Jupyter Notebooks
* **Libraries**: Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib
* **Dashboard**: Power BI / Tableau
* **Data Source**: CSV (from internal Delhivery systems or simulation)

---

## 🔐 MetaData Summary

| Field             | Description                                          |
| ----------------- | ---------------------------------------------------- |
| **Title**         | Predictive Modelling of Delhivery Logistics          |
| **Domain**        | Logistics / Transportation                           |
| **Target Column** | `actual_time`                                        |
| **Type**          | Regression Problem                                   |
| **Industry**      | Supply Chain & Last-Mile Delivery                    |
| **Data Volume**   | 144,867 rows                                         |
| **Unique Trips**  | 14,817                                               |
| **Tools**         | Python, Power BI / Tableau                           |
| **Repo Owner**    | \Technocolabs Softwraes Inc.                                      |
| **Contributors**  | \Will added shortly                             |
| **License**       | MIT / Custom                                         |

---
