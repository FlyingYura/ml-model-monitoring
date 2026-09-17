# ML Model Monitoring — Evidently Reports

Monitor the deployed asthma model in production-like conditions: data drift, target drift, and classification quality over time.

## What this project covers

- Load **reference** data (historical patients) and **current** predictions from PostgreSQL
- Build **Evidently** reports:
  - Data quality
  - Data drift
  - Target drift
  - Classification performance
- Export an HTML dashboard (`lab5_monitoring_dashboard.html`)

## Stack

`Python` · `pandas` · `SQLAlchemy` · `Evidently` · `PostgreSQL`