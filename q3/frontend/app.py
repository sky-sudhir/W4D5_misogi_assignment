import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Sales Conversion Predictor", page_icon="🤖")
st.title("📞 Sales Call Conversion Predictor")
st.markdown("""
Use this tool to predict whether a sales call transcript will likely result in a successful conversion or not.
""")

st.sidebar.header("Upload & Predict")

transcript_input = st.text_area("Enter Sales Call Transcript", height=250, placeholder="Paste your call transcript here...")

if st.button("🔍 Predict Conversion"):
    if not transcript_input.strip():
        st.warning("Please enter a transcript before predicting.")
    else:
        with st.spinner("Predicting..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={"transcript": transcript_input}
                )
                if response.status_code == 200:
                    result = response.json()
                    label = result["prediction"].capitalize()
                    confidence = result["confidence"]

                    st.success(f"Prediction: **{label}**")
                    st.progress(confidence / 100.0)
                    st.caption(f"Confidence: {confidence}%")
                else:
                    st.error("Prediction failed. Check API logs.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using FastAPI & Streamlit")
