# Churn analytics dashboard for Customer-Churn-Records.csv

## Run on your computer

    pip install -r requirements.txt
    streamlit run app.py

Only `app.py` and `requirements.txt` are needed: the Customer-Churn-Records dataset is built into `app.py`.
If a CSV named like `Customer-Churn-Records.csv` sits next to `app.py`, the app uses that instead, and the sidebar
upload box lets you analyse a different file.

## Run in Google Colab

Open `Run_dashboard_in_Colab.ipynb` in Colab and run the cells in order (steps are written inside).

## What is in it

Tabs: Overview, Segments, Geography x age, Churned vs retained, Financial stability, High-value,
Prediction & high-risk, Data. Sidebar filters: country, age, gender, credit, tenure, balance, engagement,
products, card type.

KPIs at the top (from the project report): overall churn rate, highest segment churn, high-value churn
ratio, geographic risk index, engagement drop. The balance that counts as high-value is set in the sidebar.

Prediction & high-risk tab: press the button to train five models (Logistic Regression, Decision Tree,
Random Forest, Gradient Boosting, K-Nearest Neighbors), compare them, see the confusion matrix of the
best one (with false positives and true negatives), the features that drive the score, risk tiers, and
the high-risk customer list with a CSV download. A leakage check removes any field that almost equals
the churn label (in this file, `Complain`).

Balances carry no currency in the file. Results show association, not cause.
