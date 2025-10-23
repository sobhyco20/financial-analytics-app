import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# اختيارات عامة للتطبيق
st.set_page_config(page_title="تحليلات مالية", page_icon="📊", layout="wide")

REQUIRED_BASE_COLS = {"date", "revenue", "cogs", "expenses"}  # الأعمدة الأساسية
OPTIONAL_COLS = {
    "product", "customer", "invoice_id", "vat_rate",
    "purchase_vat", "sales_vat", "ar_days", "ap_days"
}


# ========= ميزان المراجعة: دوال مساعدة =========
import re

AR_MONTHS = ["يناير","فبراير","مارس","أبريل","مايو","يونيو","يوليو","أغسطس","سبتمبر","أكتوبر","نوفمبر","ديسمبر"]

def detect_month_cols(columns):
    """اكتشاف أعمدة الأشهر مثل 'يناير/2025' أو 'مارس 2025' أو 'Jan-2025'."""
    month_cols = []
    for c in columns:
        c_str = str(c)
        # عربي (يناير/2025 أو يناير 2025)
        if any(m in c_str for m in AR_MONTHS):
            month_cols.append(c)
            continue
        # نمط عام يحتوي على شهر/سنة
        if re.search(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)|\d{1,2}/\d{4}", c_str, re.IGNORECASE):
            month_cols.append(c)
    # الترتيب: حسب ظهور الشهر العربي إن أمكن
    def month_key(c):
        txt = str(c)
        idx = None
        for i,m in enumerate(AR_MONTHS):
            if m in txt:
                idx = i
                break
        return idx if idx is not None else 99
    return sorted(month_cols, key=month_key)

def tb_guess_cols(df):
    """محاولة ذكية للتعرف على الأعمدة الشائعة في ميزان المراجعة."""
    cols = {c:str(c).strip().lower() for c in df.columns}
    inv = {v:k for k,v in cols.items()}
    def find(*names):
        for n in names:
            if n in inv: return inv[n]
        # ابحث جزئي بالعربي/إنجليزي
        for k,v in cols.items():
            if any(n in v for n in names):
                return k
        return None
    return {
        "acc_code": find("account_code","ledger account","account","رقم الحساب","كود"),
        "acc_name": find("account_name","name","اسم الحساب","البيان"),
        "opening":  find("opening","رصيد افتتاحي","الرصيد الافتتاحي"),
        "is_col":   find("قائمة الدخل","income","statement_type"),   # عمود يدل أن الحساب يتبع قائمة الدخل (اختياري)
        "bs_col":   find("المركز المالي","balance","financial position"),  # عمود يدل أن الحساب يتبع المركز المالي (اختياري)
        "op_exp":   find("التشغيلية","operating"),
        "sell_exp": find("البيعية","selling"),
        "adm_exp":  find("الادارية","الإدارية","administrative"),
    }

def tb_period_sum(df, month_cols, until_col):
    """جمع حركة الفترة من بداية السنة حتى شهر محدد."""
    if not month_cols:
        return pd.Series([0]*len(df))
    upto = month_cols[:month_cols.index(until_col)+1] if until_col in month_cols else month_cols
    return df[upto].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)

def classify_bs(name_or_label):
    txt = str(name_or_label).strip().lower()
    # كلمات مفتاحية عربية/إنجليزية
    if any(k in txt for k in ["أصول","اصل","asset","inventory","cash","bank","ذمم مدينة","مدينون","مخزون","ثابت"]):
        return "ASSETS"
    if any(k in txt for k in ["خصوم","التزامات","payable","قروض","موردين","ذمم دائنة"]):
        return "LIABILITIES"
    if any(k in txt for k in ["حقوق ملكية","equity","رأس المال","رأس المال","احتياطي","مخصص"]):
        return "EQUITY"
    return None

def classify_is(name_or_label):
    txt = str(name_or_label).strip().lower()
    if any(k in txt for k in ["مبيعات","ايراد","إيراد","revenue","sales"]):
        return "REVENUE"
    if any(k in txt for k in ["تكلفة","cogs","cost of","تكاليف المبيعات"]):
        return "COGS"
    if any(k in txt for k in ["مصروف","expense","رواتب","ايجار","إيجار","تشغيلية","بيعية","إدارية","ادارية"]):
        return "EXPENSE"
    return None

def build_financials_from_tb(df, period_amount, guesses):
    """
    df: إطار ميزان المراجعة الأصلي (أو بعد تحديد الأعمدة)
    period_amount: سلسلة تمثل مجموع الحركة YTD لكل حساب
    guesses: تخمينات الأعمدة من tb_guess_cols
    """
    # أعمدة الأسماء
    name_col = guesses["acc_name"] or df.columns[0]
    # محاولة استخراج أعمدة التصنيف إن كانت موجودة
    is_col = guesses["is_col"]
    bs_col = guesses["bs_col"]

    out = pd.DataFrame({
        "account": df.get(guesses["acc_code"], pd.Series([None]*len(df))),
        "name": df[name_col],
        "opening": pd.to_numeric(df.get(guesses["opening"], 0), errors="coerce").fillna(0),
        "ytd": pd.to_numeric(period_amount, errors="coerce").fillna(0),
    })
    out["closing"] = out["opening"] + out["ytd"]

    # ===== قائمة الدخل =====
    # مصادر: عمود "قائمة الدخل" إن وجد، أو بالاستدلال على الاسم/أعمدة التشغيل/البيع/الإدارة
    is_flag = df[is_col] if is_col in df.columns else None
    out["is_class"] = None
    if is_flag is not None:
        out.loc[:, "is_class"] = df[is_col].apply(classify_is)
    else:
        out.loc[:, "is_class"] = df[name_col].apply(classify_is)

    # تقوية التصنيف بالمصروفات التفصيلية إن وجدت
    for sub_col, label in [(guesses["op_exp"],"EXPENSE"), (guesses["sell_exp"],"EXPENSE"), (guesses["adm_exp"],"EXPENSE")]:
        if sub_col in df.columns:
            out.loc[df[sub_col].fillna(0)!=0, "is_class"] = "EXPENSE"

    revenue = out.loc[out["is_class"]=="REVENUE","ytd"].sum()
    cogs    = out.loc[out["is_class"]=="COGS","ytd"].sum()
    opex    = out.loc[out["is_class"]=="EXPENSE","ytd"].sum()
    gross_profit = revenue - cogs
    net_profit   = gross_profit - opex

    is_table = pd.DataFrame({
        "البند": ["الإيرادات","تكلفة المبيعات","الربح الإجمالي","مصروفات التشغيل/البيع/الإدارة","صافي الربح"],
        "القيمة": [revenue, -cogs, gross_profit, -opex, net_profit]
    })

    # ===== المركز المالي =====
    bs_flag = df[bs_col] if bs_col in df.columns else None
    out["bs_class"] = None
    if bs_flag is not None:
        out.loc[:, "bs_class"] = df[bs_col].apply(classify_bs)
    else:
        out.loc[:, "bs_class"] = df[name_col].apply(classify_bs)

    assets = out.loc[out["bs_class"]=="ASSETS","closing"].sum()
    liab   = out.loc[out["bs_class"]=="LIABILITIES","closing"].sum()
    equity = out.loc[out["bs_class"]=="EQUITY","closing"].sum()

    bs_table = pd.DataFrame({
        "البند": ["الأصول","الخصوم","حقوق الملكية","الخصوم + حقوق الملكية - الأصول"],
        "القيمة": [assets, liab, equity, (liab + equity - assets)]
    })

    return out, is_table, bs_table, net_profit




# --------- وظائف مساعدة ---------
@st.cache_data
def read_file(file) -> pd.DataFrame:
    if file.name.lower().endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

def ensure_base_cols(df: pd.DataFrame) -> tuple[bool, set]:
    cols = set(df.columns)
    return REQUIRED_BASE_COLS.issubset(cols), REQUIRED_BASE_COLS - cols

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # توحيد الأعمدة تاريخيًا ورقمياً
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["revenue", "cogs", "expenses"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    # حسابات أساسية
    if all(c in df.columns for c in ["revenue", "cogs", "expenses"]):
        df["gross_profit"] = df["revenue"] - df["cogs"]
        df["net_profit"]   = df["gross_profit"] - df["expenses"]
    return df

def kpi_cards(df: pd.DataFrame):
    totals = df[["revenue","cogs","expenses","gross_profit","net_profit"]].sum()
    rev = totals["revenue"]
    gp  = totals["gross_profit"]
    np_ = totals["net_profit"]
    gpm = (gp / rev * 100) if rev else 0
    npm = (np_ / rev * 100) if rev else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("إجمالي الإيرادات", f"{rev:,.0f}")
    c2.metric("إجمالي صافي الربح", f"{np_:,.0f}")
    c3.metric("هامش الربح الإجمالي %", f"{gpm:.2f}%")
    c4.metric("هامش الربح الصافي %", f"{npm:.2f}%")

def timeseries_section(df: pd.DataFrame):
    st.subheader("📈 الاتجاهات الزمنية")
    if "date" not in df.columns or df["date"].isna().all():
        st.info("لا يوجد عمود تواريخ صالح للعرض.")
        return
    period = st.selectbox("التجميع حسب", ["شهر", "ربع سنوي", "سنة"], index=0)
    dfg = df.dropna(subset=["date"]).copy()
    dfg["year"] = dfg["date"].dt.year
    if period == "شهر":
        dfg["period"] = dfg["date"].dt.to_period("M").astype(str)
    elif period == "ربع سنوي":
        dfg["period"] = dfg["date"].dt.to_period("Q").astype(str)
    else:
        dfg["period"] = dfg["date"].dt.to_period("Y").astype(str)
    agg = dfg.groupby("period")[["revenue","expenses","net_profit"]].sum().reset_index()
    fig = px.line(agg, x="period", y=["revenue","expenses","net_profit"], markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.write("**جدول التجميع:**")
    st.dataframe(agg, use_container_width=True)

def dim_analysis(df: pd.DataFrame, dim: str, top_n: int = 10):
    if dim not in df.columns:
        st.info(f"لا يوجد عمود '{dim}' في البيانات.")
        return
    st.subheader(f"تحليل حسب { 'المنتج' if dim=='product' else 'العميل' }")
    grp = df.groupby(dim)[["revenue","cogs","expenses","gross_profit","net_profit"]].sum().reset_index()
    grp["net_margin%"] = np.where(grp["revenue"]>0, grp["net_profit"]/grp["revenue"]*100, 0)
    grp_sorted = grp.sort_values("net_profit", ascending=False).head(top_n)
    fig = px.bar(grp_sorted, x=dim, y="net_profit")
    st.plotly_chart(fig, use_container_width=True)
    st.write("**أعلى العناصر ربحية:**")
    st.dataframe(grp_sorted, use_container_width=True)

def cashflow_and_aging(df: pd.DataFrame):
    st.subheader("💵 التدفقات والأعمار")

    # تدفق نقدي مبسّط: صافي تحصيل = revenue - expenses
    cf = df[["date","revenue","expenses"]].dropna(subset=["date"]).copy()
    cf["cash_flow"] = cf["revenue"] - cf["expenses"]
    by_m = cf.groupby(cf["date"].dt.to_period("M")).agg({"cash_flow":"sum"}).reset_index()
    by_m["period"] = by_m["date"].astype(str)
    fig = px.bar(by_m, x="period", y="cash_flow")
    st.plotly_chart(fig, use_container_width=True)
    st.write("**التدفق الشهري:**")
    st.dataframe(by_m[["period","cash_flow"]], use_container_width=True)

    # أعمار الذمم المدينة (AR)
    if "ar_days" in df.columns:
        st.markdown("**أعمار الذمم المدينة (AR)**")
        ar_vals = pd.to_numeric(df["ar_days"], errors="coerce")
        ar_bins = [-1,30,60,90,180,365,1_000_000]
        ar_labels = ["0-30","31-60","61-90","91-180","181-365",">365"]
        ar_bucket = pd.cut(ar_vals, bins=ar_bins, labels=ar_labels, include_lowest=True)
        ar_pivot = (
            df.assign(ar_bucket=ar_bucket)
              .groupby("ar_bucket", dropna=True)["revenue"]
              .sum()
              .reset_index(name="amount")
        )
        if len(ar_pivot):
            fig_ar = px.pie(ar_pivot, names="ar_bucket", values="amount")
            st.plotly_chart(fig_ar, use_container_width=True)
            st.dataframe(ar_pivot, use_container_width=True)
        else:
            st.info("لا توجد بيانات صالحة لأعمار الذمم المدينة.")

    # أعمار الذمم الدائنة (AP)
    if "ap_days" in df.columns:
        st.markdown("**أعمار الذمم الدائنة (AP)**")
        ap_vals = pd.to_numeric(df["ap_days"], errors="coerce")
        ap_bins = [-1,30,60,90,180,365,1_000_000]
        ap_labels = ["0-30","31-60","61-90","91-180","181-365",">365"]
        ap_bucket = pd.cut(ap_vals, bins=ap_bins, labels=ap_labels, include_lowest=True)
        ap_pivot = (
            df.assign(ap_bucket=ap_bucket)
              .groupby("ap_bucket", dropna=True)["expenses"]
              .sum()
              .reset_index(name="amount")
        )
        if len(ap_pivot):
            fig_ap = px.pie(ap_pivot, names="ap_bucket", values="amount")
            st.plotly_chart(fig_ap, use_container_width=True)
            st.dataframe(ap_pivot, use_container_width=True)
        else:
            st.info("لا توجد بيانات صالحة لأعمار الذمم الدائنة.")


def forecast_simple(df: pd.DataFrame):
    st.subheader("🔮 توقّعات مبسطة")
    if "date" not in df.columns:
        st.info("البيانات لا تحتوي على تاريخ.")
        return
    ts = df.dropna(subset=["date"]).copy()
    ts = ts.groupby(ts["date"].dt.to_period("M"))[["revenue","expenses","net_profit"]].sum().reset_index()
    ts["date"] = pd.PeriodIndex(ts["date"]).to_timestamp()
    target = st.selectbox("المؤشر", ["revenue", "expenses", "net_profit"], index=2)
    horizon = st.slider("عدد الأشهر للتوقع", 3, 12, 6)
    # Moving Average بسيط كتجربة
    ts["ma3"] = ts[target].rolling(3).mean()
    future_vals = []
    last_vals = ts[target].tail(3).tolist()
    for _ in range(horizon):
        pred = np.mean(last_vals[-3:]) if len(last_vals) >= 3 else np.mean(last_vals)
        future_vals.append(pred)
        last_vals.append(pred)
    future_index = pd.date_range(ts["date"].max() + pd.offsets.MonthBegin(), periods=horizon, freq="MS")
    fut = pd.DataFrame({"date": future_index, target: future_vals})

    fig = px.line(pd.concat([ts[["date",target]], fut], ignore_index=True), x="date", y=target)
    st.plotly_chart(fig, use_container_width=True)
    st.write("**جدول التوقعات:**")
    st.dataframe(fut, use_container_width=True)

def vat_report(df: pd.DataFrame):
    st.subheader("🧾 تقرير ضريبة القيمة المضافة (مبسّط)")
    needed = {"vat_rate"}  # يمكن أيضًا استخدام sales_vat/purchase_vat إن وُجدت
    if "sales_vat" in df.columns or "purchase_vat" in df.columns:
        sales_vat = df.get("sales_vat", pd.Series([0]*len(df)))
        purchase_vat = df.get("purchase_vat", pd.Series([0]*len(df)))
        net = sales_vat.sum() - purchase_vat.sum()
        c1,c2,c3=st.columns(3)
        c1.metric("ضريبة المخرجات (مبيعات)", f"{sales_vat.sum():,.2f}")
        c2.metric("ضريبة المدخلات (مشتريات)", f"{purchase_vat.sum():,.2f}")
        c3.metric("صافي الضريبة المستحقة", f"{net:,.2f}")
        return
    if not needed.issubset(df.columns):
        st.info("لا توجد أعمدة ضريبة صريحة. أضف 'vat_rate' أو حقول sales_vat/purchase_vat.")
        return
    # مثال: إذا لم توجد أعمدة ضريبة منفصلة، نقدّر ضريبة المخرجات = revenue * vat_rate
    # وضريبة المدخلات = expenses * vat_rate (تقدير مبسّط فقط للعرض)
    rate = pd.to_numeric(df["vat_rate"], errors="coerce").fillna(0)/100.0
    sales_vat = (df["revenue"]*rate).sum()
    purchase_vat = (df["expenses"]*rate).sum()
    net = sales_vat - purchase_vat
    c1,c2,c3=st.columns(3)
    c1.metric("ضريبة المخرجات (مبيعات)", f"{sales_vat:,.2f}")
    c2.metric("ضريبة المدخلات (مشتريات)", f"{purchase_vat:,.2f}")
    c3.metric("صافي الضريبة المستحقة", f"{net:,.2f}")

def compare_two_files():
    st.subheader("🔀 مقارنة ملفين")
    f1 = st.file_uploader("ملف 1", type=["csv","xlsx"], key="cmp1")
    f2 = st.file_uploader("ملف 2", type=["csv","xlsx"], key="cmp2")
    if f1 and f2:
        df1 = read_file(f1)
        df2 = read_file(f2)
        st.write("**ملخص عام:**")
        c1,c2 = st.columns(2)
        c1.write(f"صفوف ملف 1: {len(df1)} / أعمدة: {len(df1.columns)}")
        c2.write(f"صفوف ملف 2: {len(df2)} / أعمدة: {len(df2.columns)}")
        common_cols = list(set(df1.columns).intersection(df2.columns))
        if not common_cols:
            st.warning("لا توجد أعمدة مشتركة للمقارنة.")
            return
        key = st.selectbox("اختر عمود المفتاح للمقارنة", common_cols)
        # مقارنة مبسطة: عناصر موجودة هنا وغير موجودة هناك
        v1 = set(df1[key].dropna().astype(str))
        v2 = set(df2[key].dropna().astype(str))
        only_in_1 = sorted(list(v1 - v2))
        only_in_2 = sorted(list(v2 - v1))
        st.write("**موجود في ملف 1 فقط:**", len(only_in_1))
        st.write(only_in_1[:50])
        st.write("**موجود في ملف 2 فقط:**", len(only_in_2))
        st.write(only_in_2[:50])

def export_results(df: pd.DataFrame):
    st.subheader("⬇️ حفظ وتنزيل النتائج")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Data")
        if "date" in df.columns:
            monthly = df.dropna(subset=["date"]).copy()
            monthly["period"] = monthly["date"].dt.to_period("M").astype(str)
            (monthly.groupby("period")[["revenue","cogs","expenses","gross_profit","net_profit"]]
             .sum().reset_index()).to_excel(xw, index=False, sheet_name="Monthly")
    st.download_button("تنزيل ملف Excel شامل", buf.getvalue(), file_name="analysis_export.xlsx")

# --------- الواجهة الرئيسية ---------
st.title("📊 منصة تحليل البيانات المالية")

# تبويب: رفع/اختيار ملف
tab_data, tab_overview, tab_ts, tab_dim_prod, tab_dim_cust, tab_cash, tab_fcst, tab_vat, tab_cmp, tab_export = st.tabs([
    "📂 ملف البيانات", "📌 نظرة عامة", "⏱️ تحليل زمني",
    "🛒 المنتجات", "👤 العملاء", "💵 التدفقات والأعمار",
    "🔮 التوقّعات", "🧾 ضريبة القيمة المضافة", "🔀 مقارنة ملفين", "⬇️ حفظ/تنزيل"
])

with tab_data:
    st.markdown("**ارفع ملف CSV/Excel بالأعمدة الأساسية:** `date, revenue, cogs, expenses`")
    uploaded = st.file_uploader("ارفع الملف هنا", type=["csv","xlsx"], key="main_upl")
    if uploaded:
        try:
            df_raw = read_file(uploaded)
            ok, missing = ensure_base_cols(df_raw)
            if not ok:
                st.error(f"الأعمدة الناقصة: {', '.join(missing)}")
            else:
                st.success("✅ تم تحميل الملف بنجاح.")
                st.session_state["df"] = normalize_df(df_raw)
                st.dataframe(st.session_state["df"].head(50), use_container_width=True)
                extra = set(df_raw.columns) - REQUIRED_BASE_COLS
                if extra:
                    st.info("أعمدة إضافية متاحة للتحليل: " + ", ".join([c for c in extra if c in OPTIONAL_COLS]))
        except Exception as e:
            st.error(f"تعذر قراءة الملف: {e}")

if "df" not in st.session_state:
    st.warning("◀️ ارفع ملفك في تبويب (ملف البيانات) أولًا لاستخدام بقية التبويبات.")
else:
    df = st.session_state["df"]

    with tab_overview:
        st.subheader("نظرة عامة على الأداء")
        kpi_cards(df)
        if "date" in df.columns and not df["date"].isna().all():
            by_m = df.dropna(subset=["date"]).copy()
            by_m["period"] = by_m["date"].dt.to_period("M").astype(str)
            agg = by_m.groupby("period")[["revenue","expenses","net_profit"]].sum().reset_index()
            fig = px.area(agg, x="period", y=["revenue","expenses","net_profit"])
            st.plotly_chart(fig, use_container_width=True)

    with tab_ts:
        timeseries_section(df)

    with tab_dim_prod:
        dim_analysis(df, "product")

    with tab_dim_cust:
        dim_analysis(df, "customer")

    with tab_cash:
        cashflow_and_aging(df)

    with tab_fcst:
        forecast_simple(df)

    with tab_vat:
        vat_report(df)

    with tab_cmp:
        compare_two_files()

    with tab_export:
        export_results(df)
