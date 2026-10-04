import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")

st.title("🎓 Student Performance Prediction & Recommendation")

st.sidebar.header("Student Details")
age = st.sidebar.slider("Age", 15, 22, 17)
studytime = st.sidebar.slider("Study time (1=<2h, 4=>10h)", 1, 4, 2)
failures = st.sidebar.slider("Past failures", 0, 3, 0)
absences = st.sidebar.slider("Absences", 0, 60, 5)
G1 = st.sidebar.slider("First period grade (G1)", 0, 20, 10)
G2 = st.sidebar.slider("Second period grade (G2)", 0, 20, 10)
goout = st.sidebar.slider("Going out (1-5)", 1, 5, 3)
health = st.sidebar.slider("Health (1-5)", 1, 5, 3)

if st.button("Predict"):
    # all columns 0, then fill the ones we have
    row = pd.DataFrame([[0] * len(columns)], columns=columns)
    inputs = {"age": age, "studytime": studytime, "failures": failures,
              "absences": absences, "G1": G1, "G2": G2,
              "goout": goout}
    for k, v in inputs.items():
        if k in row.columns:
            row[k] = v
    for h in ["health", "health_status"]:
        if h in row.columns:
            row[h] = health

    pred = model.predict(row)[0]
    if pred == 1:
        st.success("✅ Prediction: PASS")
    else:
        st.error("❌ Prediction: AT RISK (Fail)")

    st.subheader("📚 Recommendations")
    tips = []
    if absences > 10:
        tips.append("Attendance improve pannunga, absences romba adhigam.")
    if studytime <= 1:
        tips.append("Daily study time increase pannunga (minimum 2 hours).")
    if failures > 0:
        tips.append("Pazhaya failed subjects ku extra practice + tutor help edunga.")
    if G1 < 10 or G2 < 10:
        tips.append("Weak subjects ku YouTube / Khan Academy videos + previous papers practice pannunga.")
    if goout >= 4:
        tips.append("Outing / social time konjam kammi pannunga.")
    if not tips:
        tips.append("Super! Ippadiye continue pannunga 👍")
    for t in tips:
        st.write("•", t)
