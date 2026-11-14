"""Customer Segmentation & Churn Pattern Analytics dashboard.

Built for Customer-Churn-Records.csv (columns: RowNumber, CustomerId, Surname, CreditScore, Geography,
Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Exited, Complain,
Satisfaction Score, Card Type, Point Earned).

Run:  pip install -r requirements.txt
      streamlit run app.py
Data: put Customer-Churn-Records.csv next to this file, or upload it in the sidebar.
The "Prediction & high-risk" tab trains five models inside the app (press the button there).
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
    fig.add_hline(y=overall, line_dash="dash", annotation_text=f"overall {overall:.1f}%", annotation_position="top left")
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



@st.cache_data(show_spinner=False)
def fit_models(data: pd.DataFrame, seed: int = 42) -> dict:
    """Leakage check, five models on an 80/20 split, permutation importance, out-of-fold risk scores."""
    from scipy.stats import chi2_contingency
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
    from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.tree import DecisionTreeClassifier

    y = data.Churn
    drop = [c for c in ["Churn", "Exited", "CustomerId", "RowNumber", "AgeSegment", "CreditBand", "TenureGroup",
                        "BalanceSegment", "SalaryQuartile", "Engagement"] if c in data]
    X = data.drop(columns=drop)

    # leakage check: a field that almost equals the answer would make every model look perfect
    rows = []
    for c in X.columns:
        col = X[c]
        if col.dtype.kind in "if" and col.nunique() > 10:
            a = roc_auc_score(y, col)
            score, kind, limit = max(a, 1 - a), "single-feature AUC", 0.95
        else:
            ct = pd.crosstab(col, y)
            if ct.shape[0] < 2:
                continue
            score = float(np.sqrt(chi2_contingency(ct)[0] / (len(y) * (min(ct.shape) - 1))))
            kind, limit = "Cramers V", 0.90
        rows.append({"Feature": c, "Measure": kind, "Score": round(float(score), 3), "Leaky": bool(score > limit)})
    leak = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    X = X.drop(columns=leak.loc[leak.Leaky, "Feature"].tolist())

    num = X.select_dtypes(include=[np.number]).columns.tolist()
    cat = X.select_dtypes(exclude=[np.number]).columns.tolist()

    def pre(scale):
        return ColumnTransformer([("num", StandardScaler() if scale else "passthrough", num),
                                  ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])

    models = {
        "Logistic Regression": Pipeline([("prep", pre(True)), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))]),
        "Decision Tree": Pipeline([("prep", pre(False)), ("clf", DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=30, class_weight="balanced", random_state=seed))]),
        "Random Forest": Pipeline([("prep", pre(False)), ("clf", RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced", n_jobs=-1, random_state=seed))]),
        "Gradient Boosting": Pipeline([("prep", pre(False)), ("clf", GradientBoostingClassifier(random_state=seed))]),
        "K-Nearest Neighbors": Pipeline([("prep", pre(True)), ("clf", KNeighborsClassifier(n_neighbors=25))]),
    }
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    test_prob, auc, pr_auc, roc = {}, {}, {}, {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        test_prob[name] = p
        auc[name] = float(roc_auc_score(yte, p))
        pr_auc[name] = float(average_precision_score(yte, p))
        fpr, tpr, _ = roc_curve(yte, p)
        roc[name] = (fpr, tpr)
    best = max(auc, key=auc.get)

    imp = permutation_importance(models[best], Xte, yte, scoring="roc_auc", n_repeats=5, random_state=seed, n_jobs=-1)
    importance = (pd.DataFrame({"Feature": X.columns, "Importance": imp.importances_mean})
                  .sort_values("Importance", ascending=False).reset_index(drop=True))

    # honest scores for every customer: each is scored by a model that never saw that customer
    from sklearn.base import clone
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = cross_val_predict(clone(models[best]), X, y, cv=cv, method="predict_proba")[:, 1]

    # risk drivers: segments whose churn rate is at least 1.25 times the average (and at least 100 customers)
    base = y.mean()
    drivers = pd.Series("", index=X.index)
    for c in X.columns:
        col = X[c]
        lab = pd.qcut(col, 4, duplicates="drop").astype(str) if (col.dtype.kind in "if" and col.nunique() > 8) else col.astype(str)
        g = y.groupby(lab).agg(["mean", "size"])
        risky = set(g[(g["mean"] >= 1.25 * base) & (g["size"] >= 100)].index)
        m_ = lab.isin(risky)
        drivers[m_] = drivers[m_] + c + "=" + lab[m_] + "; "
    drivers = drivers.str.rstrip("; ").replace("", "none")

    return {"leak": leak, "features": X.columns.tolist(), "best": best, "auc": auc, "pr_auc": pr_auc, "roc": roc,
            "test_prob": test_prob, "y_test": yte.to_numpy(), "importance": importance,
            "oof": pd.Series(oof, index=X.index), "drivers": drivers}


def confusion(y_true, prob, thr):
    pred = prob >= thr
    tp = int(((pred == 1) & (y_true == 1)).sum()); fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum()); fn = int(((pred == 0) & (y_true == 1)).sum())
    return tn, fp, fn, tp


def prf(tn, fp, fn, tp):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


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
     "High-value", "Prediction & high-risk", "Data"]
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

# ---------------------------------------------------------------- prediction and high-risk customers
with tab_risk:
    st.caption("Five models are trained on 80% of all customers and scored on the other 20%. The sidebar filters do not change "
               "the models; they only choose which customers are listed below.")
    if st.button("Train the five models (about a minute the first time)", key="train"):
        st.session_state["models_on"] = True
    if not st.session_state.get("models_on"):
        st.info("Press the button to train the models, compare them, and build the high-risk customer list.")
    else:
        try:
            with st.spinner("Training models..."):
                res = fit_models(full)
        except ImportError:
            st.error("scikit-learn is missing. Run: pip install scikit-learn")
            st.stop()
        leaky = res["leak"].loc[res["leak"].Leaky, "Feature"].tolist()
        if leaky:
            st.warning(f"Leakage check removed {', '.join(leaky)}: it is almost the same field as the churn label, so keeping it "
                       "would make every model look perfect without being useful.")
        else:
            st.success("Leakage check: no field gives away the answer.")
        with st.expander("Leakage check details"):
            st.dataframe(res["leak"].head(8), width="stretch", hide_index=True)

        best = res["best"]
        y_t = res["y_test"]
        st.subheader("1. Model comparison")
        thr = st.slider("Decision threshold (flag a customer as leaving when the probability is at least this)", 0.05, 0.95, 0.40, 0.05)
        comp = []
        for name, p in res["test_prob"].items():
            tn, fp, fn, tp = confusion(y_t, p, thr)
            prec, rec, f1 = prf(tn, fp, fn, tp)
            comp.append({"Model": name, "Test AUC": res["auc"][name], "PR-AUC": res["pr_auc"][name], "Precision": prec,
                         "Recall": rec, "F1": f1, "Accuracy": (tn + tp) / len(y_t), "False positives": fp, "False negatives": fn})
        comp = pd.DataFrame(comp).sort_values("Test AUC", ascending=False)
        st.dataframe(comp.round(3), width="stretch", hide_index=True)
        st.caption(f"Best model by test AUC: {best} ({res['auc'][best]:.3f}). AUC measures how well a model ranks leavers above stayers "
                   "(0.5 is guessing, 1.0 is perfect). Accuracy can mislead because most customers stay.")
        roc = pd.concat([pd.DataFrame({"False positive rate": f, "True positive rate": t, "Model": n}) for n, (f, t) in res["roc"].items()])
        fig = px.line(roc, x="False positive rate", y="True positive rate", color="Model", title="ROC curves on the held-back customers")
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
        st.plotly_chart(fig, width="stretch", key=next_key())

        st.subheader(f"2. Confusion matrix: {best}")
        tn, fp, fn, tp = confusion(y_t, res["test_prob"][best], thr)
        prec, rec, f1 = prf(tn, fp, fn, tp)
        cc1, cc2 = st.columns([2, 3])
        z = pd.DataFrame([[tn, fp], [fn, tp]], index=["Actually stayed", "Actually left"], columns=["Predicted stay", "Predicted leave"])
        cc1.plotly_chart(px.imshow(z, text_auto=True, color_continuous_scale="Blues", title=f"Threshold {thr:.2f}"),
                         width="stretch", key=next_key())
        with cc2:
            q1, q2 = st.columns(2)
            q1.metric("True negatives", f"{tn:,}", "stayed, left alone", delta_color="off")
            q2.metric("False positives", f"{fp:,}", f"{fp / max(fp + tn, 1) * 100:.1f}% of stayers wrongly flagged", delta_color="off")
            q1.metric("False negatives", f"{fn:,}", "left, missed", delta_color="off")
            q2.metric("True positives", f"{tp:,}", f"recall {rec * 100:.1f}%", delta_color="off")
            st.caption(f"Specificity {tn / max(tn + fp, 1) * 100:.1f}%  |  Precision {prec * 100:.1f}%  |  F1 {f1:.3f}. "
                       "A lower threshold catches more leavers but wastes more offers on stayers.")
        trows = []
        for t_ in np.arange(0.10, 0.91, 0.10):
            a_, b_, c_, d_ = confusion(y_t, res["test_prob"][best], t_)
            p_, r_, f_ = prf(a_, b_, c_, d_)
            trows.append({"Threshold": round(t_, 2), "Flagged": b_ + d_, "TN": a_, "FP": b_, "FN": c_, "TP": d_,
                          "Precision": round(p_, 3), "Recall": round(r_, 3), "F1": round(f_, 3)})
        st.dataframe(pd.DataFrame(trows), width="stretch", hide_index=True)

        st.subheader("3. What drives the risk score")
        imp = res["importance"].head(10).sort_values("Importance")
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                     title="Permutation importance: how much the model's AUC drops when a feature is shuffled")
        fig.update_traces(marker_color=BLUE)
        st.plotly_chart(fig, width="stretch", key=next_key())

        st.subheader("4. Risk tiers and high-risk customers")
        t1, t2 = st.columns(2)
        hi_share = t1.slider("High-risk share of customers", 5, 40, 20, 5) / 100
        med_share = t2.slider("Medium-risk share", 10, 50, 30, 5) / 100
        sc = full.copy()
        sc["ChurnProbability"] = res["oof"].round(4)
        rk = sc.ChurnProbability.rank(pct=True, ascending=False, method="first")
        sc["RiskTier"] = np.where(rk <= hi_share, "High", np.where(rk <= hi_share + med_share, "Medium", "Low"))
        sc["ExpectedBalanceAtRisk"] = (sc.ChurnProbability * sc.Balance).round(0)
        sc["RiskDrivers"] = res["drivers"]
        tier = sc.groupby("RiskTier").agg(Customers=("Churn", "size"), Churned=("Churn", "sum"), ChurnRate=("Churn", "mean"))
        tier = tier.reindex(["High", "Medium", "Low"])
        tier["Lift"] = tier.ChurnRate / sc.Churn.mean()
        tier["% of all churners"] = tier.Churned / tier.Churned.sum() * 100
        tier["ChurnRate"] *= 100
        tier = tier.reset_index()
        g1, g2 = st.columns([3, 2])
        g1.plotly_chart(bar(tier, "RiskTier", rate(sc.Churn), "Actual churn rate by predicted risk tier"), width="stretch", key=next_key())
        g2.dataframe(tier.rename(columns={"ChurnRate": "Actual churn %"}).round(2), width="stretch", hide_index=True)
        st.caption("If the model works, actual churn rises from Low to High. Lift 2 means twice the average churn rate.")

        listing = sc.loc[d.index]
        listing = listing[listing.RiskTier == "High"].sort_values("ExpectedBalanceAtRisk", ascending=False)
        cols_out = [c for c in ["CustomerId", "Geography", "Gender", "Age", "Tenure", "CreditScore", "Balance", "EstimatedSalary",
                                "NumOfProducts", "HasCrCard", "IsActiveMember", "Satisfaction Score", "Card Type", "Point Earned",
                                "ChurnProbability", "RiskTier", "ExpectedBalanceAtRisk", "RiskDrivers"] if c in listing]
        out = listing[cols_out].assign(ActualChurn=listing.Churn)
        out.insert(0, "PriorityRank", range(1, len(out) + 1))
        r1, r2, r3 = st.columns(3)
        r1.metric("High-risk customers in the current filters", f"{len(out):,}")
        r2.metric("Expected balance at risk", f"{out.ExpectedBalanceAtRisk.sum() / 1e6:,.1f}M")
        r3.metric("Of them, actually left", f"{out.ActualChurn.mean() * 100:.1f}%" if len(out) else "n/a")
        st.dataframe(out.head(500), width="stretch", hide_index=True)
        st.download_button("Download high-risk customers (CSV)", out.to_csv(index=False).encode(),
                           "high_risk_customers.csv", "text/csv")
        st.caption("Risk drivers are the known risk segments a customer belongs to. They describe association, not cause.")

# ---------------------------------------------------------------- data
with tab_data:
    st.dataframe(d.head(500), width="stretch", hide_index=True)
    st.download_button("Download filtered data (CSV)", d.to_csv(index=False).encode(), "filtered_customers.csv", "text/csv")
