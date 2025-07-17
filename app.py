import streamlit as st
import pandas as pd
from main import get_top_positive_employees, get_top_negative_employees, get_flight_risk_employees

st.title("Employee Sentiment Dashboard")

if st.button("Show Top Employees"):
    top_pos = get_top_positive_employees()
    top_neg = get_top_negative_employees()
    st.subheader("Top Positive Employees")
    st.dataframe(top_pos)
    st.subheader("Top Negative Employees")
    st.dataframe(top_neg)

if st.button("Show Flight Risks"):
    risks = get_flight_risk_employees()
    st.subheader("Flight Risk Employees")
    st.write(risks)
