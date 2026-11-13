"""Customer Segmentation & Churn Pattern Analytics dashboard.

Built for Customer-Churn-Records.csv (columns: RowNumber, CustomerId, Surname, CreditScore, Geography,
Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited, Complain,
Satisfaction Score, Card Type, Point Earned).

Run:  pip install -r requirements.txt
      streamlit run app.py
Data: put Customer-Churn-Records.csv next to this file, or upload it in the sidebar.
Optional: put high_risk_customers.csv (from the notebook, Step 27) next to this file for the High-risk tab.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Churn Analytics", layout="wide")

HERE = Path(__file__).parent
AGE_ORDER = ["<30", "30-45", "46-60", "60+"]
CREDIT_ORDER = ["Low (<580)", "Medium (580-739)", "High (740+)"]
TENURE_ORDER = ["New (0-2y)", "Mid-term (3-6y)", "Long-term (7y+)"]
BAL_ORDER = ["Zero", "Low (1-50k)", "Medium (50-100k)", "High (100k+)"]
SAL_ORDER = ["Q1 lowest", "Q2", "Q3", "Q4 highest"]
BLUE, GREY = "#2a78d6", "#b9b8b0"
_counter = {"n": 0}


def next_key() -> str:
    """Unique key per chart so identical charts on different tabs do not collide."""
    _counter["n"] += 1
    return f"chart_{_counter['n']}"


@st.cache_data
def load(src) -> pd.DataFrame:
    df = pd.read_csv(src)
    if "Churn" not in df.columns:
        if "Exited" not in df.columns:
            raise ValueError("The file needs an 'Exited' (or 'Churn') column with 0/1 values.")
        df["Churn"] = df["Exited"]
    df = df.drop(columns=[c for c in ["Surname", "RowNumber"] if c in df.columns])
    df["AgeSegment"] = pd.cut(df.Age, [0, 29, 45, 60, 200], labels=AGE_ORDER)
    df["CreditBand"] = pd.cut(df.CreditScore, [0, 579, 739, 900], labels=CREDIT_ORDER)
    df["TenureGroup"] = pd.cut(df.Tenure, [-1, 2, 6, 100], labels=TENURE_ORDER)
    df["BalanceSegment"] = pd.cut(df.Balance, [-1, 0, 50000, 100000, 1e12], labels=BAL_ORDER)
    df["SalaryQuartile"] = pd.qcut(df.EstimatedSalary, 4, labels=SAL_ORDER)
    df["Engagement"] = np.where(df.IsActiveMember == 1, "Active", "Inactive")
    df["Gender"] = df.Gender.astype(str)
    return df


def rate(s: pd.Series) -> float:
    return float(s.mean() * 100) if len(s) else float("nan")


def seg_table(d: pd.DataFrame, col: str, total_churn: int) -> pd.DataFrame:
    g = d.groupby(col, observed=True).agg(
        Customers=("Churn", "size"), Churned=("Churn", "sum"),
        ChurnRate=("Churn", "mean"), AvgBalance=("Balance", "mean"),
    )
    g["BalanceLost"] = d[d.Churn == 1].groupby(col, observed=True).Balance.sum()
    g["BalanceLost"] = g.BalanceLost.fillna(0)
    g["ChurnRate"] *= 100
    g["ShareOfCustomers"] = g.Customers / len(d) * 100
    g["ShareOfChurners"] = g.Churned / max(total_churn, 1) * 100
    return g.reset_index()


def bar(tbl: pd.DataFrame, col: str, overall: float, title: str):
    t = tbl.copy()
    t[col] = t[col].astype(str)
    t["Above average"] = np.where(t.ChurnRate > overall, "Above overall rate", "At or below overall rate")
    fig = px.bar(
        t, x=col, y="ChurnRate", color="Above average",
        color_discrete_map={"Above overall rate": BLUE, "At or below overall rate": GREY},
        hover_data={"Customers": True, "Churned": True, "ChurnRate": ":.1f", "Above average": False},
        labels={"ChurnRate": "Churn rate (%)"}, title=title,
    )
    fig.add_hline(y=overall, line_dash="dash", annotation_text=f"overall {overall:.1f}%")
    fig.update_layout(showlegend=False, margin=dict(t=50, b=10), yaxis_rangemode="tozero")
    return fig


def heat(pt: pd.DataFrame, cnt: pd.DataFrame, min_n: int, title: str):
    shown = pt.where(cnt >= min_n)
    fig = px.imshow(shown, text_auto=".1f", aspect="auto", color_continuous_scale="Blues",
                    labels=dict(color="Churn %"), title=title)
    return fig


def grid(d: pd.DataFrame, row: str, col: str, order=None):
    pt = d.pivot_table(index=row, columns=col, values="Churn", aggfunc="mean", observed=True).mul(100)
    cnt = d.pivot_table(index=row, columns=col, values="Churn", aggfunc="size", observed=True).fillna(0)
    if order:
        keep = [c for c in order if c in pt.columns]
        pt, cnt = pt.reindex(columns=keep), cnt.reindex(columns=keep)
    return pt, cnt


# ---------------------------------------------------------------- data
_counter["n"] = 0
st.sidebar.title("Data")
upload = st.sidebar.file_uploader("Upload Customer-Churn-Records.csv", type="csv")
candidates = list(HERE.glob("Customer-Churn-Records.csv")) + list(HERE.glob("data.csv")) + list(HERE.glob("Customer*Churn*.csv"))
default = next(iter(candidates), None)
try:
    if upload is not None:
        full = load(upload)
    elif default is not None:
        full = load(default)
    else:
        st.error("No data found. Put Customer-Churn-Records.csv next to app.py, or upload it in the sidebar.")
        st.stop()
except (ValueError, KeyError) as e:
    st.error(f"This file does not look like Customer-Churn-Records.csv: {e}")
    st.stop()

# ---------------------------------------------------------------- filters
st.sidebar.title("Filters")
f = full
countries = st.sidebar.multiselect("Country", sorted(full.Geography.unique()), default=sorted(full.Geography.unique()))
ages = st.sidebar.multiselect("Age segment", AGE_ORDER, default=AGE_ORDER)
genders = st.sidebar.multiselect("Gender", sorted(full.Gender.unique()), default=sorted(full.Gender.unique()))
credit = st.sidebar.multiselect("Credit band", CREDIT_ORDER, default=CREDIT_ORDER)
tenure = st.sidebar.multiselect("Tenure group", TENURE_ORDER, default=TENURE_ORDER)
balseg = st.sidebar.multiselect("Balance segment", BAL_ORDER, default=BAL_ORDER)
eng = st.sidebar.multiselect("Engagement", ["Active", "Inactive"], default=["Active", "Inactive"])
prods = st.sidebar.multiselect("Products held", sorted(full.NumOfProducts.unique()), default=sorted(full.NumOfProducts.unique()))
mask = (
    f.Geography.isin(countries) & f.AgeSegment.isin(ages) & f.Gender.isin(genders)
    & f.CreditBand.isin(credit) & f.TenureGroup.isin(tenure) & f.BalanceSegment.isin(balseg)
    & f.Engagement.isin(eng) & f.NumOfProducts.isin(prods)
)
if "Card Type" in full:
    cards = st.sidebar.multiselect("Card type", sorted(full["Card Type"].astype(str).unique()),
                                   default=sorted(full["Card Type"].astype(str).unique()))
    mask &= f["Card Type"].astype(str).isin(cards)
st.sidebar.title("KPI settings")
HV_LINE = st.sidebar.number_input("High-value customer: balance above", 0, 1_000_000, 100_000, 10_000)
d = f[mask]

st.title("Customer Segmentation & Churn Pattern Analytics")
st.caption(f"{len(d):,} of {len(full):,} customers selected. Balances carry no currency in the source file. "
           "Everything shown is association, not cause.")
if d.empty:
    st.warning("No customers match these filters. Widen a filter in the sidebar.")
    st.stop()

overall = rate(d.Churn)
base_overall = rate(full.Churn)
total_churn = int(d.Churn.sum())

# ---------------------------------------------------------------- KPIs (the five from the brief)
hv = d[d.Balance > HV_LINE]
hv_ratio = rate(hv.Churn) / overall if len(hv) and overall > 0 else float("nan")

geo = seg_table(d, "Geography", total_churn)
geo["Index"] = geo.ChurnRate / overall * 100 if overall > 0 else np.nan
geo_ok = geo[geo.Customers >= 30]

act, ina = d[d.Engagement == "Active"].Churn, d[d.Engagement == "Inactive"].Churn
eng_ratio = (ina.mean() / act.mean()) if len(act) and len(ina) and act.mean() > 0 else float("nan")

seg_dims = ["Geography", "Gender", "AgeSegment", "CreditBand", "TenureGroup", "BalanceSegment", "Engagement"]
rows = []
for c in seg_dims:
    t = seg_table(d, c, total_churn)
    t = t[t.Customers >= 100]
    for _, r in t.iterrows():
        rows.append((f"{c}: {r[c]}", r.ChurnRate))
top_seg = max(rows, key=lambda x: x[1]) if rows else None

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Overall churn rate", f"{overall:.1f}%",
          f"{overall - base_overall:+.1f} pts vs all customers" if len(d) != len(full) else None, delta_color="off")
k2.metric("Highest segment churn", f"{top_seg[1]:.1f}%" if top_seg else "n/a", top_seg[0] if top_seg else None, delta_color="off")
k3.metric("High-value churn ratio", f"{hv_ratio:.2f}x" if hv_ratio == hv_ratio else "n/a",
          f"{len(hv):,} customers above {HV_LINE:,}", delta_color="off")
if len(geo_ok):
    g_top = geo_ok.loc[geo_ok.Index.idxmax()]
    k4.metric("Geographic risk index (highest)", f"{g_top.Index:.0f}", f"{g_top.Geography} (100 = average)", delta_color="off")
else:
    k4.metric("Geographic risk index (highest)", "n/a")
k5.metric("Engagement drop", f"{eng_ratio:.1f}x" if eng_ratio == eng_ratio else "n/a", "inactive vs active churn rate", delta_color="off")

lost_total = d[d.Churn == 1].Balance.sum()
m1, m2, m3 = st.columns(3)
m1.metric("Customers churned", f"{total_churn:,}")
m2.metric("Balance lost", f"{lost_total / 1e6:,.1f}M")
m3.metric("Share of all balances", f"{lost_total / max(d.Balance.sum(), 1) * 100:.1f}%")

tab_over, tab_seg, tab_geo, tab_profile, tab_fin, tab_hv, tab_risk, tab_data = st.tabs(
    ["Overview", "Segments", "Geography x age", "Churned vs retained", "Financial stability",
     "High-value", "High-risk customers", "Data"]
)

# ---------------------------------------------------------------- overview
with tab_over:
    c1, c2 = st.columns(2)
    for col, ttl, holder in [
        ("Geography", "Churn rate by country", c1), ("AgeSegment", "Churn rate by age segment", c2),
        ("CreditBand", "Churn rate by credit band", c1), ("Engagement", "Churn rate by engagement", c2),
        ("BalanceSegment", "Churn rate by balance segment", c1), ("TenureGroup", "Churn rate by tenure group", c2),
    ]:
        with holder:
            st.plotly_chart(bar(seg_table(d, col, total_churn), col, overall, ttl), width="stretch", key=next_key())

# ---------------------------------------------------------------- segments
with tab_seg:
    dims = {
        "Country": "Geography", "Gender": "Gender", "Age segment": "AgeSegment", "Credit band": "CreditBand",
        "Tenure group": "TenureGroup", "Balance segment": "BalanceSegment", "Salary quartile": "SalaryQuartile",
        "Engagement": "Engagement", "Products held": "NumOfProducts", "Has credit card": "HasCrCard",
    }
    for label, c in [("Complaint filed", "Complain"), ("Satisfaction score", "Satisfaction Score"), ("Card type", "Card Type")]:
        if c in d:
            dims[label] = c
    pick = st.selectbox("Segment by", list(dims))
    col = dims[pick]
    t = seg_table(d, col, total_churn)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(bar(t, col, overall, f"Churn rate by {pick.lower()}"), width="stretch", key=next_key())
    mix = t.melt(id_vars=col, value_vars=["ShareOfCustomers", "ShareOfChurners"], var_name="Measure", value_name="Percent")
    mix[col] = mix[col].astype(str)
    mix["Measure"] = mix.Measure.map({"ShareOfCustomers": "Share of customers", "ShareOfChurners": "Share of churners"})
    fig = px.bar(mix, x=col, y="Percent", color="Measure", barmode="group",
                 color_discrete_map={"Share of customers": GREY, "Share of churners": BLUE},
                 title="Contribution to churn vs size", labels={"Percent": "%"})
    fig.update_layout(margin=dict(t=50, b=10), legend_title_text="")
    c2.plotly_chart(fig, width="stretch", key=next_key())
    show = t.rename(columns={"ChurnRate": "Churn rate (%)", "AvgBalance": "Avg balance", "BalanceLost": "Balance lost",
                             "ShareOfCustomers": "% of customers", "ShareOfChurners": "% of churners"})
    st.dataframe(show.round(1), width="stretch", hide_index=True)
    small = t[t.Customers < 100]
    if len(small):
        st.caption("Segments with fewer than 100 customers are unreliable: " + ", ".join(map(str, small[col])))

# ---------------------------------------------------------------- geography x age
with tab_geo:
    min_n = st.slider("Hide cells with fewer customers than", 0, 200, 30, 10)
    piv, cnt = grid(d, "Geography", "AgeSegment", AGE_ORDER)
    st.plotly_chart(heat(piv, cnt, min_n, "Churn rate (%) by country and age segment"), width="stretch", key=next_key())
    st.caption("Customers per cell")
    st.dataframe(cnt.astype(int), width="stretch")
    spread = piv.where(cnt >= min_n)
    gap = pd.DataFrame({"Highest %": spread.max(axis=1), "Lowest %": spread.min(axis=1)})
    gap["Gap (points)"] = gap["Highest %"] - gap["Lowest %"]
    gap["Age segment with highest churn"] = spread.idxmax(axis=1)
    st.caption("How much the age pattern differs inside each country")
    st.dataframe(gap.round(1), width="stretch")
    g = d.pivot_table(index="Geography", columns="Gender", values="Churn", aggfunc="mean", observed=True).mul(100)
    st.caption("Churn rate (%) by country and gender")
    st.dataframe(g.round(1), width="stretch")
    risk = geo[["Geography", "Customers", "Churned", "ChurnRate", "Index", "ShareOfChurners"]].rename(
        columns={"ChurnRate": "Churn rate (%)", "Index": "Geographic Risk Index", "ShareOfChurners": "% of churners"})
    st.caption("Geographic Risk Index = country churn rate / overall churn rate x 100 (100 = average)")
    st.dataframe(risk.round(1), width="stretch", hide_index=True)

# ---------------------------------------------------------------- churned vs retained
with tab_profile:
    cols = [c for c in ["Age", "Tenure", "CreditScore", "Balance", "EstimatedSalary", "NumOfProducts", "IsActiveMember",
                        "HasCrCard", "Complain", "Satisfaction Score", "Point Earned"] if c in d]
    if "Complain" in d and (d.Complain == d.Churn).mean() > 0.99:
        st.warning("Complain matches Exited in over 99% of rows, so it is almost the same field as the target. "
                   "Treat its numbers with care and keep it out of any prediction model.")
    prof = d.groupby("Churn")[cols].mean().T
    if set(prof.columns) == {0, 1}:
        prof.columns = ["Retained", "Churned"]
        prof["Difference %"] = (prof.Churned / prof.Retained.replace(0, np.nan) - 1) * 100
        st.dataframe(prof.round(2), width="stretch")
        try:
            from scipy.stats import mannwhitneyu
            ps = {c: mannwhitneyu(d[d.Churn == 1][c], d[d.Churn == 0][c]).pvalue for c in cols}
            st.caption("Mann-Whitney p-values (below 0.05 = difference unlikely to be chance): "
                       + ", ".join(f"{k} {v:.3g}" for k, v in ps.items()))
        except Exception:
            pass
        cats = [c for c in ["Geography", "Gender", "Engagement", "NumOfProducts", "Card Type"] if c in d]
        parts = []
        for c in cats:
            x = pd.crosstab(d[c], d.Churn, normalize="columns") * 100
            x.columns = ["Retained %", "Churned %"]
            x["Gap (points)"] = x["Churned %"] - x["Retained %"]
            x.index = [f"{c} = {i}" for i in x.index]
            parts.append(x)
        st.caption("Share of each group in every category")
        st.dataframe(pd.concat(parts).round(1), width="stretch")
    else:
        st.info("The current filters contain only churned or only retained customers.")
    var = st.selectbox("Distribution of", cols, index=cols.index("Age") if "Age" in cols else 0)
    fig = px.histogram(d.assign(Status=d.Churn.map({0: "Retained", 1: "Churned"})), x=var, color="Status", barmode="overlay",
                       histnorm="percent", opacity=0.65, color_discrete_map={"Retained": GREY, "Churned": BLUE})
    st.plotly_chart(fig, width="stretch", key=next_key())

# ---------------------------------------------------------------- financial stability
with tab_fin:
    st.caption("Financial standing read from credit score, balance and products held.")
    min_f = st.slider("Hide cells with fewer customers than", 0, 200, 30, 10, key="fin_min")
    p1, c1_ = grid(d, "CreditBand", "BalanceSegment", BAL_ORDER)
    st.plotly_chart(heat(p1, c1_, min_f, "Churn rate (%) by credit band and balance segment"), width="stretch", key=next_key())
    p2, c2_ = grid(d, "CreditBand", "NumOfProducts")
    st.plotly_chart(heat(p2, c2_, min_f, "Churn rate (%) by credit band and products held"), width="stretch", key=next_key())
    p3, c3_ = grid(d, "Engagement", "NumOfProducts")
    st.plotly_chart(heat(p3, c3_, min_f, "Churn rate (%) by engagement and products held"), width="stretch", key=next_key())

# ---------------------------------------------------------------- high-value
with tab_hv:
    hb = d[d.Balance > HV_LINE]
    lost = d[d.Churn == 1].Balance.sum()
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("Customers above the line", f"{len(hb):,}")
    h2.metric("Churn rate", f"{rate(hb.Churn):.1f}%" if len(hb) else "n/a")
    h3.metric("Balance lost", f"{hb[hb.Churn == 1].Balance.sum() / 1e6:,.1f}M")
    h4.metric("Share of all balance lost", f"{hb[hb.Churn == 1].Balance.sum() / lost * 100:.1f}%" if lost else "n/a")
    st.caption(f"The line is set in the sidebar (now {HV_LINE:,}).")

    big = d[(d.Churn == 1)].sort_values("Balance", ascending=False)
    keep = [c for c in ["CustomerId", "Geography", "Age", "Balance", "EstimatedSalary", "NumOfProducts", "Engagement"] if c in big]
    st.caption("The 15 largest balances among customers who left")
    st.dataframe(big[keep].head(15), width="stretch", hide_index=True)

    sample = d.sample(min(len(d), 4000), random_state=1).assign(Status=lambda x: x.Churn.map({0: "Retained", 1: "Churned"}))
    fig = px.scatter(sample, x="EstimatedSalary", y="Balance", color="Status", opacity=0.55,
                     color_discrete_map={"Retained": GREY, "Churned": BLUE}, title="Salary vs balance (sample of up to 4,000)")
    st.plotly_chart(fig, width="stretch", key=next_key())
    s1, s2 = st.columns(2)
    s1.plotly_chart(bar(seg_table(d, "SalaryQuartile", total_churn), "SalaryQuartile", overall, "Churn by salary quartile"),
                    width="stretch", key=next_key())
    s2.plotly_chart(bar(seg_table(d, "BalanceSegment", total_churn), "BalanceSegment", overall, "Churn by balance segment"),
                    width="stretch", key=next_key())
    pq, cq = grid(d, "SalaryQuartile", "BalanceSegment", BAL_ORDER)
    st.caption("Churn rate (%) by salary quartile and balance segment")
    st.dataframe(pq.round(1), width="stretch")

    st.subheader("Revenue at risk")
    st.caption("The file has no revenue or profit field, so money at risk means the account balance that left. "
               "The margin below is an assumption for illustration only; set it to your bank's real figure.")
    margin = st.slider("Assumed yearly income as % of balance (illustration)", 0.0, 10.0, 2.0, 0.5)
    rv = []
    for c in ["Geography", "Engagement", "BalanceSegment"]:
        t = seg_table(d, c, total_churn)[[c, "Customers", "Churned", "ChurnRate", "BalanceLost"]]
        t.columns = ["Segment", "Customers", "Churned", "Churn rate (%)", "Balance lost"]
        t.insert(0, "Dimension", c)
        t["Segment"] = t.Segment.astype(str)
        t["% of balance lost"] = t["Balance lost"] / max(lost, 1) * 100
        t["Illustrative yearly income at risk"] = t["Balance lost"] * margin / 100
        rv.append(t)
    st.dataframe(pd.concat(rv).round(1), width="stretch", hide_index=True)

# ---------------------------------------------------------------- high-risk customers (from the notebook)
with tab_risk:
    st.caption("This tab shows the output of the notebook (Part D, Step 27). It does not train a model itself.")
    risk_up = st.file_uploader("Upload high_risk_customers.csv or all_customer_risk_scores.csv", type="csv", key="risk_up")
    risk_file = risk_up if risk_up is not None else next(
        iter(list(HERE.glob("all_customer_risk_scores.csv")) + list(HERE.glob("high_risk_customers.csv"))), None)
    if risk_file is None:
        st.info("Run the notebook through Step 27, then put high_risk_customers.csv (and optionally "
                "all_customer_risk_scores.csv) next to app.py, or upload it above.")
    else:
        rk = pd.read_csv(risk_file)
        if "RiskTier" in rk:
            hi_ = rk[rk.RiskTier == "High"]
            st.caption(f"{len(rk):,} customers scored; {len(hi_):,} are in the High tier.")
            if "ActualChurn" in rk:
                tier = rk.groupby("RiskTier").agg(Customers=("ActualChurn", "size"), Churned=("ActualChurn", "sum"),
                                                  ChurnRate=("ActualChurn", "mean"))
                tier["ChurnRate"] *= 100
                tier = tier.reindex([t for t in ["High", "Medium", "Low"] if t in tier.index]).reset_index()
                st.plotly_chart(bar(tier, "RiskTier", rate(rk.ActualChurn), "Actual churn rate by predicted risk tier"),
                                width="stretch", key=next_key())
        else:
            hi_ = rk
        k_a, k_b, k_c = st.columns(3)
        k_a.metric("High-risk customers", f"{len(hi_):,}")
        if "ExpectedBalanceAtRisk" in hi_:
            k_b.metric("Expected balance at risk", f"{hi_.ExpectedBalanceAtRisk.sum() / 1e6:,.1f}M")
        if "ActualChurn" in hi_ and len(hi_):
            k_c.metric("Actually left", f"{hi_.ActualChurn.mean() * 100:.1f}%")
        if "Geography" in hi_:
            gsel = st.multiselect("Country", sorted(hi_.Geography.unique()), default=sorted(hi_.Geography.unique()), key="risk_geo")
            hi_ = hi_[hi_.Geography.isin(gsel)]
        st.dataframe(hi_.head(500), width="stretch", hide_index=True)
        st.download_button("Download this list (CSV)", hi_.to_csv(index=False).encode(), "high_risk_customers_filtered.csv", "text/csv")

# ---------------------------------------------------------------- data
with tab_data:
    st.dataframe(d.head(500), width="stretch", hide_index=True)
    st.download_button("Download filtered data (CSV)", d.to_csv(index=False).encode(), "filtered_customers.csv", "text/csv")
