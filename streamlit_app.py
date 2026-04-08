import streamlit as st
import requests
from datetime import datetime

st.set_page_config(page_title="FraudShield AI", layout="wide", page_icon="🛡️")
st.title("🛡️ Insurance Fraud Detection System")

def handle_unknown(value):
    return "?" if value == "Unknown" else value

with st.form("main_form"):
    st.subheader("📋 Policy & Customer Profile")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        age = st.number_input("Insured Age", 18, 100, 35)
        months_as_customer = st.number_input("Months as Customer", 0, 600, 120)
        policy_state = st.selectbox("Policy State", ["OH", "IL", "IN"])
    with col2:
        policy_csl = st.selectbox("Policy CSL", ["100/300", "250/500", "500/1000"])
        policy_deductable = st.selectbox("Policy Deductable", [500, 1000, 2000])
        umbrella_limit = st.number_input("Umbrella Limit", value=0)
    with col3:
        insured_sex = st.selectbox("Insured Sex", ["MALE", "FEMALE"])
        insured_education = st.selectbox("Education Level", ["JD", "High School", "Associate", "MD", "Masters", "PhD", "College"])
        insured_occupation = st.selectbox("Occupation", ["machine-op-inspct", "prof-specialty", "tech-support", "sales", "exec-managerial", "craft-repair", "transport-moving", "priv-house-serv", "other-service", "armed-forces", "adm-clerical", "protective-serv", "handlers-cleaners", "farming-fishing"])
    with col4:
        insured_relationship = st.selectbox("Relationship", ["husband", "other-relative", "own-child", "unmarried", "wife", "not-in-family"])
        insured_hobbies = st.selectbox("Hobbies", ["reading", "exercise", "basketball", "paintball", "chess", "skydiving", "camping", "movies", "golf", "board-games", "bungie-jumping", "base-jumping", "yachting", "hiking", "video-games", "polo", "kayaking", "dancing", "cross-fit", "sleeping"])
        policy_bind_date = st.date_input("Policy Bind Date", datetime(2014, 1, 1))

    st.subheader("💥 Incident Information")
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        incident_type = st.selectbox("Incident Type", ["Single Vehicle Collision", "Multi-vehicle Collision", "Side Collision", "Parked Car"])
        collision_type = st.selectbox("Collision Type", ["Rear Collision", "Side Collision", "Front Collision", "Unknown"])
        incident_severity = st.selectbox("Incident Severity", ["Trivial Damage", "Minor Damage", "Major Damage", "Total Loss"])
    with col6:
        authorities_contacted = st.selectbox("Authorities", ["Police", "Fire", "Other", "None", "Ambulance"])
        incident_state = st.selectbox("Incident State", ["SC", "VA", "NY", "OH", "WV", "NC", "PA"])
        incident_city = st.selectbox("Incident City", ["Columbus", "Riverport", "Arlington", "Springfield", "Hillsdale", "Northbend", "Northbrook"])
    with col7:
        incident_date = st.date_input("Incident Date", datetime(2015, 1, 1))
        incident_hour = st.slider("Hour of Day", 0, 23, 12)
        number_of_vehicles = st.number_input("Vehicles Involved", 1, 10, 1)
    with col8:
        property_damage = st.selectbox("Property Damage", ["NO", "YES", "Unknown"])
        police_report = st.selectbox("Police Report", ["NO", "YES", "Unknown"])
        witnesses = st.slider("Witnesses", 0, 3, 0)
        bodily_injuries = st.slider("Bodily Injuries", 0, 2, 0)

    st.subheader("💰 Financial & Vehicle Data")
    col9, col10, col11, col12 = st.columns(4)
    with col9:
        capital_gains = st.number_input("Capital Gains", value=0)
        capital_loss = st.number_input("Capital Loss", value=0)
    with col10:
        total_claim_amount = st.number_input("Total Claim ($)", value=5000)
        injury_claim = st.number_input("Injury Claim ($)", value=500)
    with col11:
        property_claim = st.number_input("Property Claim ($)", value=500)
        vehicle_claim = st.number_input("Vehicle Claim ($)", value=4000)
    with col12:
        auto_make = st.selectbox("Auto Make", ["Saab", "Mercedes", "Dodge", "Chevrolet", "Accura", "Nissan", "Audi", "Toyota", "Ford", "Hyundai", "BMW", "Mazda", "Volkswagen", "Jeep"])
        auto_year = st.number_input("Auto Year", 1995, 2026, 2012)
        auto_model = st.text_input("Auto Model", "RAM")

    submit = st.form_submit_button("🔍 RUN FRAUD ANALYSIS")

if submit:
    payload = {
        "age": age, "months_as_customer": months_as_customer,
        "policy_bind_date": str(policy_bind_date), "policy_state": policy_state,
        "policy_csl": policy_csl, "policy_deductable": policy_deductable,
        "umbrella_limit": umbrella_limit, "capital_gains": capital_gains,
        "capital_loss": capital_loss, "insured_sex": insured_sex,
        "insured_education_level": insured_education, "insured_occupation": insured_occupation,
        "insured_hobbies": insured_hobbies, "insured_relationship": insured_relationship,
        "incident_date": str(incident_date), "incident_type": incident_type,
        "collision_type": handle_unknown(collision_type), "incident_severity": incident_severity,
        "authorities_contacted": authorities_contacted, "incident_state": incident_state,
        "incident_city": incident_city, "incident_hour_of_the_day": incident_hour,
        "number_of_vehicles_involved": number_of_vehicles,
        "property_damage": handle_unknown(property_damage), "bodily_injuries": bodily_injuries,
        "witnesses": witnesses, "police_report_available": handle_unknown(police_report),
        "total_claim_amount": total_claim_amount, "injury_claim": injury_claim,
        "property_claim": property_claim, "vehicle_claim": vehicle_claim,
        "auto_make": auto_make, "auto_model": auto_model, "auto_year": auto_year
    }

    with st.spinner("Analyzing..."):
        try:
            response = requests.post("http://127.0.0.1:8000/predict", json=payload)
            if response.status_code == 200:
                result = response.json().get('prediction')
                if result == "Fraudulent":
                    st.error(f"### 🚩 Prediction: {result}")
                else:
                    st.success(f"### ✅ Prediction: {result}")
            else:
                st.error(f"Error: {response.text}")
        except Exception as e:
            st.error(f"Connection Failed: {e}")
            