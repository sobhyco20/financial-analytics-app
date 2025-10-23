import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="تحليل البيانات المالية", page_icon="💰", layout="wide")

st.title("💰 تطبيق تحليل البيانات المالية")
st.write("قم برفع ملف Excel أو CSV يحتوي على الأعمدة: `date`, `revenue`, `cogs`, `expenses`")

uploaded_file = st.file_uploader("📤 ارفع ملف البيانات", type=["csv", "xlsx"])

if uploaded_file:
    try:
        # قراءة الملف
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        # التأكد من الأعمدة
        expected_cols = {"date", "revenue", "cogs", "expenses"}
        if not expected_cols.issubset(df.columns):
            st.error(f"الملف يجب أن يحتوي على الأعمدة: {', '.join(expected_cols)}")
        else:
            st.success("✅ تم تحميل البيانات بنجاح!")

            # الحسابات
            df["gross_profit"] = df["revenue"] - df["cogs"]
            df["net_profit"] = df["gross_profit"] - df["expenses"]

            totals = df[["revenue", "cogs", "expenses", "gross_profit", "net_profit"]].sum()

            col1, col2, col3 = st.columns(3)
            col1.metric("إجمالي الإيرادات", f"{totals['revenue']:,.0f}")
            col2.metric("صافي الربح", f"{totals['net_profit']:,.0f}")
            margin = (totals['net_profit'] / totals['revenue'] * 100) if totals['revenue'] else 0
            col3.metric("هامش الربح الصافي", f"{margin:.2f}%")

            # الرسوم
            st.subheader("📈 الإيرادات والمصروفات عبر الوقت")
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                chart = px.line(df, x="date", y=["revenue", "expenses", "net_profit"], markers=True)
                st.plotly_chart(chart, use_container_width=True)

            # تصدير النتائج
            st.subheader("📤 تحميل النتائج")
            result_file = "financial_results.xlsx"
            df.to_excel(result_file, index=False)
            with open(result_file, "rb") as f:
                st.download_button("⬇️ تحميل الملف النهائي", f, file_name=result_file)

    except Exception as e:
        st.error(f"حدث خطأ أثناء معالجة الملف: {e}")
