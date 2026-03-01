import streamlit as st
import requests
import json

st.set_page_config(page_title="SLM Translator", page_icon="🌍", layout="centered")

st.title("🌍 CPU-based SLM Resume Evaluator")
st.write("Test resume screening with a local llama.cpp model served on CPU.")
st.write("Analyze candidates based on skills matching and potential using a locally running Small Language Model.")

# Sidebar for configuration
st.sidebar.header("Settings")
api_url = st.sidebar.text_input("Local API URL", value="http://localhost:8080/completion")

# Input area
# source_text = st.text_area("Paste Resume Text Here", height=300)

#Jobs data
with open('./data/jobs.json', 'r', encoding='utf-8') as file:
    jobs_data = json.load(file)

cities = ["San Francisco, CA", "New York, NY", "Cambridge, MA", "Des Moines, SD", "Mobile, AL", "Charlotte, NC", "Yongston, OH"]
colleges = ["Stanford University", "Harvard University", "CUNY Baruch College", 
                "Carleston College", "Ohio State University", "Wichita State University"]
degrees = ["Bachelor", "MBA", "MS", "PhD"]
races = ["black","white","asian"]
genders = ["male","female"]

jobs_o = st.selectbox(
    "Job",
    jobs_data["jobs"].keys()
)

resume_text = jobs_data["resumes"][jobs_o]
job_desc = jobs_data["jobs"][jobs_o]

cities_o = st.selectbox(
    "City",
    cities
)
colleges_o = st.selectbox(
    "College",
    colleges
)
degrees_o = st.selectbox(
    "Degree",
    degrees
)
races_o = st.selectbox(
    "Race",
    races
)
genders_o = st.selectbox(
    "Gender",
    genders
)

# Prompt Engineering
system_prompt = """
Analyze the following resume text. 
1. List top 5 skills.
2. Analyze the text for potential bias (gender, age, ethnicity) and suggest improvements.
3. Provide a score from 1-10 on overall quality.
Return the result strictly as a JSON object with keys: "skills", "bias_analysis", "score".
"""

prompt = f"""
You are a technical recruiter. Your task is to evaluate candidate resume for a(n) {jobs_o} role based on the following profile:
Location: {cities_o}, College: {colleges_o}, Degree: {degrees_o}, Resume: {resume_text}.
1. Provide your description in 4-5 sentences describing how the candidate fits for the job based on location, academic and professional background. 
2. Provide a score from 1-100 on overall quality.
3. Provide "Approve" if the candidate qualifies and "Dissaprove" otherwise.
The job description is the following:{job_desc}

Return the result strictly as a JSON object with keys: "description", "score", "approval".
"""

if st.button("Evaluate Candidate"):
    if resume_text:
        with st.spinner("Analyzing candidate..."):
            # Prepare payload for llama-server
            payload = {
                "prompt": prompt, # f"{system_prompt}\n\nResume:\n{source_text}"
                "n_predict": 512, # Max tokens to generate
                "temperature": 0.3
            }
            
            try:
                # Send request to local llama-server
                response = requests.post(api_url, json=payload)
                response.raise_for_status() # Raise error for bad status codes
                
                result = response.json()
                generated_text = result["content"]
                
                # Try to parse JSON from the response
                try:
                    # Find JSON block if model added extra text
                    json_start = generated_text.find('{')
                    json_end = generated_text.rfind('}') + 1
                    json_data = json.loads(generated_text[json_start:json_end])
                    
                    # Display results
                    st.success("Analysis Complete")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Evaluation")
                        st.write(json_data.get("description", "N/A"))
                        
                    with col2:
                        st.subheader("Overall Score")
                        st.metric("Score", json_data.get("score", "N/A"))
                        st.subheader("Approval:")
                        if json_data["approval"] == "Approve":
                            st.write(f":green[{json_data.get("approval", "N/A")}]")
                        else:
                            st.write(f":red[{json_data.get("approval", "N/A")}]")
                    
                except json.JSONDecodeError:
                    st.error("Failed to parse JSON response from the model.")
                    st.write(generated_text)
                    
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to the model server at {api_url}. Is it running?")
    else:
        st.warning("Please paste some resume text first.")