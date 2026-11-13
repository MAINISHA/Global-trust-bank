# Churn analytics dashboard for Customer-Churn-Records.csv

    pip install -r requirements.txt
    streamlit run app.py

Put `Customer-Churn-Records.csv` in the same folder as `app.py` (or upload it from the sidebar).

For the **High-risk customers** tab, run the notebook through Step 27 and put `high_risk_customers.csv`
(and optionally `all_customer_risk_scores.csv`) in the same folder, or upload it in that tab.

Tabs: Overview, Segments, Geography x age, Churned vs retained, Financial stability, High-value,
High-risk customers, Data.

KPIs at the top: overall churn rate, highest segment churn, high-value churn ratio, geographic risk index,
engagement drop. The high-value balance line is set in the sidebar (default 100,000).

Segments: age <30 / 30-45 / 46-60 / 60+; credit Low <580 / Medium 580-739 / High 740+;
tenure New 0-2 / Mid 3-6 / Long 7+ years; balance Zero / Low 1-50k / Medium 50-100k / High 100k+.

Balances carry no currency in the file. Results show association, not cause.
