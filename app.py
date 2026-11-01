
import streamlit as st
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# Page Configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)
# Welcome Screen
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:

    st.title("🎓 Welcome to Student Performance Prediction System")
    st.markdown(
    """
    <div style="background: linear-gradient(120deg, #172554, #4338ca);
                padding: 25px; border-radius: 15px; color: white;">
        <h2>Smart Academic Monitoring</h2>
        <p>Identify academic risks early and support better student outcomes.</p>
    </div>
    """,
    unsafe_allow_html=True
)

    st.subheader("Identify Academic Risk. Support Better Learning.")

    st.write(
        "This Machine Learning application helps identify students "
        "who may need additional academic support."
    )

    st.markdown("---")

    st.subheader("✨ What You Can Explore")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("📊 **Dashboard**")
        st.caption("Explore student performance and analytics.")

    with col2:
        st.write("🤖 **Risk Prediction**")
        st.caption("Predict High Risk or Low Risk.")

    with col3:
        st.write("📈 **Visual Reports**")
        st.caption("View charts and performance metrics.")

    st.markdown("---")

    st.info("Built using Python, Machine Learning and Streamlit.")

    if st.button("🚀 Get Started", type="primary"):
        st.session_state.started = True
        st.rerun()

    st.stop()

# Load Model and Dataset
model = joblib.load("student_risk_model.pkl")
df = pd.read_csv("student_data.csv")

# Sidebar
st.sidebar.title("🎓 Student Risk System")
st.sidebar.write("AI-Based Academic Monitoring")

st.sidebar.info(
    "This system predicts student academic risk "
    "using attendance, marks, assignments and study habits."
)

st.sidebar.caption(
    "Developed using Python, Pandas, Scikit-learn and Streamlit"
)

st.sidebar.divider()
st.sidebar.subheader("🧭 Navigation")

st.sidebar.markdown("[🏠 Dashboard](#student-performance-prediction-system)")

st.sidebar.markdown("[🤖 Student Prediction](#predict-student-academic-risk)")

st.sidebar.markdown("[ℹ️ About Project](#about-this-project)")
st.sidebar.subheader("Project Features")

st.sidebar.write("✔ Student Risk Prediction")
st.sidebar.write("✔ Model Performance Evaluation")
st.sidebar.write("✔ Confusion Matrix")
st.sidebar.write("✔ Academic Risk Distribution")
st.sidebar.write("✔ Attendance Analysis")
st.sidebar.write("✔ Student Report Download")

# Main Heading
st.title("🎓 Student Performance Prediction System")

st.write(
    "AI-powered Academic Risk Prediction and Early Warning Dashboard"
)

st.divider()

# Dashboard Metrics
col1, col2, col3 = st.columns(3)

col1.metric("Total Students", len(df))

col2.metric(
    "High Risk Students",
    len(df[df["academic_risk"] == "High Risk"])
)

col3.metric(
    "Low Risk Students",
    len(df[df["academic_risk"] == "Low Risk"])
)

st.divider()

# Model Performance Evaluation
X = df.drop(["student_id", "academic_risk"], axis=1)
y = df["academic_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred,
    pos_label="High Risk",
    zero_division=0
)

recall = recall_score(
    y_test, y_pred,
    pos_label="High Risk",
    zero_division=0
)

f1 = f1_score(
    y_test, y_pred,
    pos_label="High Risk",
    zero_division=0
)

st.subheader("📊 Model Performance Evaluation")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Accuracy", f"{accuracy * 100:.2f}%")
c2.metric("Precision", f"{precision * 100:.2f}%")
c3.metric("Recall", f"{recall * 100:.2f}%")
c4.metric("F1 Score", f"{f1 * 100:.2f}%")

# Confusion Matrix
st.subheader("🎯 Confusion Matrix")

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["Low Risk", "High Risk"]
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Low Risk", "Actual High Risk"],
    columns=["Predicted Low Risk", "Predicted High Risk"]
)

st.dataframe(cm_df)

st.divider()

# Student Risk Prediction
st.header("🔍 Predict Student Academic Risk")

with st.form("student_form"):

    col1, col2 = st.columns(2)

    with col1:
        attendance = st.slider(
            "Attendance (%)", 0, 100, 75
        )

        internal_marks = st.slider(
            "Internal Marks", 0, 100, 60
        )

        assignment_completion = st.slider(
            "Assignment Completion (%)", 0, 100, 70
        )

    with col2:
        previous_semester_gpa = st.slider(
            "Previous Semester GPA", 0.0, 10.0, 7.0
        )

        study_hours = st.slider(
            "Daily Study Hours", 0.0, 8.0, 3.0
        )

        class_participation = st.slider(
            "Class Participation", 1, 10, 5
        )

    submit = st.form_submit_button("Predict Academic Risk")

if submit:

    input_data = pd.DataFrame([{
        "attendance": attendance,
        "internal_marks": internal_marks,
        "assignment_completion": assignment_completion,
        "previous_semester_gpa": previous_semester_gpa,
        "study_hours": study_hours,
        "class_participation": class_participation
    }])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    confidence = max(probability) * 100

    st.metric(
        "Prediction Confidence",
        f"{confidence:.2f}%"
    )

    # Download Individual Prediction Report
    report_data = input_data.copy()

    report_data["Predicted Risk"] = prediction
    report_data["Confidence"] = f"{confidence:.2f}%"

    st.download_button(
        label="Download Prediction Report",
        data=report_data.to_csv(index=False),
        file_name="student_prediction_report.csv",
        mime="text/csv"
    )

    # Prediction Result
    st.subheader("Prediction Result")

    if prediction == "High Risk":

        st.error("⚠️ High Academic Risk Detected")

        st.write(
            "This student may benefit from additional academic support."
        )

        st.write("Suggested Actions:")
        st.write("- Provide extra classes.")
        st.write("- Monitor attendance.")
        st.write("- Help complete pending assignments.")

    else:

        st.success("✅ Low Academic Risk")

        st.write(
            "The student currently shows relatively positive academic indicators."
        )

    # Important Prediction Factors
    st.subheader("📊 Important Prediction Factors")

    st.write(
        "The model identifies important factors that influence academic risk."
    )

    importance = pd.DataFrame({
        "Factor": input_data.columns,
        "Importance": model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    st.bar_chart(importance.set_index("Factor"))

st.divider()

# Academic Risk Distribution
st.header("📊 Academic Risk Distribution")

risk_counts = df["academic_risk"].value_counts()

st.bar_chart(risk_counts)

st.divider()

# Student Attendance Analysis
st.header("📈 Student Attendance Analysis")

st.line_chart(df["attendance"].head(50))

st.divider()

# Student Dataset Preview
st.header("📋 Student Dataset Preview")

risk_filter = st.selectbox(
    "Filter Students by Risk",
    ["All Students", "High Risk", "Low Risk"]
)

if risk_filter == "All Students":
    filtered_df = df
else:
    filtered_df = df[
        df["academic_risk"] == risk_filter
    ]

st.dataframe(
    filtered_df.head(20),
    use_container_width=True
)

st.divider()

# Download Student Report
st.header("📥 Download Student Report")

csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Student Data",
    data=csv,
    file_name="student_performance_report.csv",
    mime="text/csv"
)
st.divider()

st.header("ℹ️ About This Project")

st.write("""
The Student Performance Prediction and Early Warning System
is a Machine Learning-based application designed to identify
students who may need additional academic support.

It uses attendance, internal marks, assignment completion,
previous semester GPA, study hours and class participation
to predict academic risk.
""")

st.subheader("Technologies Used")

st.write("Python | Pandas | Scikit-learn | Streamlit")

st.subheader("Project Purpose")

st.info(
    "This project is developed for educational and learning purposes. "
    "It uses a synthetic dataset and is not validated for real-world "
    "academic decisions."
)