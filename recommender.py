"""
JobRecommender — Core ML Module
"""

import pandas as pd
import numpy as np
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class JobRecommender:

    def __init__(self):
        self.df = None
        self.tfidf = None
        self.tfidf_matrix = None
        self.is_trained = False

    # ─────────────────────────────────────────────────────
    # 1. DATA PREPROCESSING
    # ─────────────────────────────────────────────────────

    def _clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = str(text).lower()
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[^a-z0-9\s\+\#]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def _build_combined_text(self, row) -> str:
        desc = self._clean_text(row.get("description", ""))
        skills = self._clean_text(row.get("skills_desc", ""))
        title = self._clean_text(row.get("title", ""))
        return f"{title} {desc} {skills} {skills} {skills}"

    def _normalize_salary(self, row) -> float:
        if not pd.isna(row.get("normalized_salary")):
            return float(row["normalized_salary"])

        if not pd.isna(row.get("max_salary")) and not pd.isna(row.get("min_salary")):
            mid = (float(row["max_salary"]) + float(row["min_salary"])) / 2
            period = str(row.get("pay_period", "YEARLY")).upper()
            multipliers = {"HOURLY": 2080, "MONTHLY": 12, "YEARLY": 1}
            return mid * multipliers.get(period, 1)

        return np.nan

    # ─────────────────────────────────────────────────────
    # 2. TRAINING
    # ─────────────────────────────────────────────────────

    def train(self, csv_path: str):
        print(f"[INFO] Loading dataset from {csv_path} ...")
        df = pd.read_csv(csv_path, low_memory=False)

        df = df.dropna(subset=["description", "title"])

        print("[INFO] Building text corpus ...")
        df["combined_text"] = df.apply(self._build_combined_text, axis=1)
        df["annual_salary"] = df.apply(self._normalize_salary, axis=1)

        df["company_name"] = df["company_name"].fillna("Unknown Company")
        df["skills_desc"] = df["skills_desc"].fillna("Not specified")
        df["location"] = df["location"].fillna("Not specified")
        df["formatted_experience_level"] = df.get("formatted_experience_level", pd.Series()).fillna("Not specified")

        self.df = df.reset_index(drop=True)

        print("[INFO] Fitting TF-IDF vectorizer ...")
        self.tfidf = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
            sublinear_tf=True
        )

        self.tfidf_matrix = self.tfidf.fit_transform(self.df["combined_text"])
        self.is_trained = True
        print(f"[INFO] Training complete. Matrix shape: {self.tfidf_matrix.shape}")

    # ─────────────────────────────────────────────────────
    # 3. RECOMMENDATION (FIXED 🔥)
    # ─────────────────────────────────────────────────────

    def recommend(self, user_skills: str, top_n: int = 10) -> list:
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        query = self._clean_text(user_skills)
        query_vec = self.tfidf.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = scores.argsort()[::-1][:top_n]
        results = []

        for rank, idx in enumerate(top_indices, start=1):
            row = self.df.iloc[idx]

            # ✅ FIX: Handle NaN properly
            experience = row.get("formatted_experience_level")
            if pd.isna(experience):
                experience = "Not specified"

            salary = row["annual_salary"]
            if pd.isna(salary):
                salary = None

            results.append({
                "rank": rank,
                "match_score": round(float(scores[idx]), 4),
                "job_title": str(row.get("title", "Not specified")),
                "company": str(row.get("company_name", "Unknown Company")),
                "location": str(row.get("location", "Not specified")),
                "work_type": str(row.get("formatted_work_type", "Full-time")),
                "experience_level": experience,   # ✅ FIXED
                "annual_salary_usd": round(float(salary), 2) if salary else None,
                "required_skills": str(row.get("skills_desc", "Not specified"))[:300],
                "job_url": str(row.get("job_posting_url", "")),
            })

        return results

    # ─────────────────────────────────────────────────────
    # 4. SALARY BENCHMARK
    # ─────────────────────────────────────────────────────

    def salary_benchmark(self, job_title: str) -> dict:
        mask = self.df["title"].str.contains(job_title, case=False, na=False)
        subset = self.df[mask & self.df["annual_salary"].notna()]["annual_salary"]

        if subset.empty:
            return {"message": f"No salary data found for '{job_title}'"}

        return {
            "count": int(len(subset)),
            "min": round(float(subset.min()), 2),
            "median": round(float(subset.median()), 2),
            "mean": round(float(subset.mean()), 2),
            "max": round(float(subset.max()), 2),
        }

    # ─────────────────────────────────────────────────────
    # 5. TOP COMPANIES
    # ─────────────────────────────────────────────────────

    def top_companies(self, job_title: str, top_n: int = 10) -> list:
        mask = self.df["title"].str.contains(job_title, case=False, na=False)
        subset = self.df[mask & (self.df["company_name"] != "Unknown Company")]
        counts = subset["company_name"].value_counts().head(top_n)
        return [{"company": c, "job_openings": int(n)} for c, n in counts.items()]

    # ─────────────────────────────────────────────────────
    # 6. SKILLS FOR ROLE
    # ─────────────────────────────────────────────────────

    def skills_for_role(self, job_title: str, top_n: int = 20) -> list:
        mask = self.df["title"].str.contains(job_title, case=False, na=False)
        subset = self.df[mask]

        all_text = " ".join(
            subset["skills_desc"].fillna("").tolist() +
            subset["description"].fillna("").tolist()
        ).lower()

        keywords = ["python", "java", "sql", "machine learning", "aws", "docker"]

        skill_counts = {}
        for skill in keywords:
            count = len(re.findall(r"\b" + skill + r"\b", all_text))
            if count > 0:
                skill_counts[skill] = count

        sorted_skills = sorted(skill_counts.items(), key=lambda x: -x[1])[:top_n]
        return [{"skill": s, "count": c} for s, c in sorted_skills]

    # ─────────────────────────────────────────────────────
    # 7. SAVE / LOAD
    # ─────────────────────────────────────────────────────

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "df": self.df,
                "tfidf": self.tfidf,
                "tfidf_matrix": self.tfidf_matrix,
            }, f)
        print(f"[INFO] Model saved to {path}")

    def load(self, path: str):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        self.df = obj["df"]
        self.tfidf = obj["tfidf"]
        self.tfidf_matrix = obj["tfidf_matrix"]
        self.is_trained = True
        print(f"[INFO] Model loaded from {path}")