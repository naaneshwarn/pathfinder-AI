from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import re
import PyPDF2
from recommender import JobRecommender

app = Flask(__name__)
CORS(app)

MODEL_PATH = "models/job_recommender.pkl"
DATA_PATH = "C:/Users/Naaneshwar.K/OneDrive/Desktop/LinkedIn_Job_Posting/postings.csv"

recommender = JobRecommender()

if os.path.exists(MODEL_PATH):
    recommender.load(MODEL_PATH)
else:
    recommender.train(DATA_PATH)
    recommender.save(MODEL_PATH)

# -------------------------------
# READ FILE (PDF/TXT)
# -------------------------------
def extract_text(file):
    if file.filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    else:
        return file.read().decode("utf-8")


# -------------------------------
# SMART ROLE DETECTION
# -------------------------------
def detect_role(text):
    text = text.lower()

    if "machine learning" in text or "ai" in text:
        return "machine learning"
    elif "data" in text:
        return "data scientist"
    elif "react" in text or "frontend" in text:
        return "frontend"
    elif "backend" in text or "api" in text:
        return "backend"
    else:
        return "software developer"


# -------------------------------
# ROADMAP GENERATOR
# -------------------------------
def get_roadmap(role):
    roadmaps = {
        "data scientist": [
            "Python + Statistics",
            "Pandas, NumPy",
            "Machine Learning",
            "Visualization",
            "Projects",
            "SQL"
        ],
        "machine learning": [
            "Python",
            "Math",
            "ML Algorithms",
            "Deep Learning",
            "Projects",
            "Deploy Models"
        ],
        "frontend": [
            "HTML, CSS",
            "JavaScript",
            "React",
            "Projects",
            "APIs"
        ],
        "backend": [
            "Python/Node",
            "Database",
            "APIs",
            "Auth",
            "Deployment"
        ],
        "software developer": [
            "DSA",
            "Programming",
            "Projects",
            "System Design",
            "Apply Jobs"
        ]
    }
    return roadmaps.get(role, roadmaps["software developer"])


# -------------------------------
# RESUME ANALYZER (🔥 MAIN)
# -------------------------------
@app.route("/analyze-resume", methods=["POST"])
def analyze_resume():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    text = extract_text(file).lower()

    skills_db = [
        "python","java","sql","machine learning","deep learning",
        "data analysis","react","node","aws","docker",
        "communication","teamwork"
    ]

    found_skills = [s for s in skills_db if s in text]

    strengths = []
    weaknesses = []

    if len(found_skills) > 5:
        strengths.append("Strong technical skills")
    if "project" in text:
        strengths.append("Project experience present")
    if "internship" in text:
        strengths.append("Industry exposure")

    if len(found_skills) < 3:
        weaknesses.append("Add more technical skills")
    if "project" not in text:
        weaknesses.append("Add projects")
    if "certification" not in text:
        weaknesses.append("Add certifications")

    # 🔥 SMART ROLE
    role = detect_role(text)

    # 🔥 JOB RECOMMENDATION
    jobs = recommender.recommend(" ".join(found_skills), 5)

    # 🔥 ROADMAP
    roadmap = get_roadmap(role)

    return jsonify({
        "role_detected": role,
        "skills": found_skills,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "jobs": jobs,
        "roadmap": roadmap
    })


# -------------------------------
# CHAT
# -------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg = data.get("message", "")

    return jsonify(recommender.recommend(msg, 5))


# -------------------------------
# RUN
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)