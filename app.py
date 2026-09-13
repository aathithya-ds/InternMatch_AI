from flask import Flask, render_template, request, jsonify
from recommender import InternshipRecommender
from skill_gap import SkillGapAnalyzer

app = Flask(__name__)

# Initialize AI modules
recommender = InternshipRecommender()
skill_analyzer = SkillGapAnalyzer()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def recommend():

    data = request.get_json()

    student_id = data.get("student_id", "S001")
    top_n = int(data.get("top_n", 5))

    recommendations = recommender.recommend(
        student_id,
        top_n=top_n
    )

    results = []

    for _, row in recommendations.iterrows():

        skill_info = skill_analyzer.analyze(
            student_id,
            row["internship_id"]
        )

        results.append({
            "internship_id": row["internship_id"],
            "title": row["title"],
            "company": row["company"],
            "domain": row["domain"],
            "experience_level": row["experience_level"],
            "duration": row["duration"],
            "work_mode": row["work_mode"],
            "stipend": int(row["stipend"]),
            "match_score": round(
                float(row["match_score"]),
                2
            ),
            "matched_skills": skill_info["matched_skills"],
            "missing_skills": skill_info["missing_skills"],
            "skill_match": skill_info["match_percentage"]
        })

    return jsonify({
        "student_id": student_id,
        "recommendations": results
    })


@app.route("/students")
def students():

    data = recommender.students[
        [
            "student_id",
            "name",
            "skills",
            "domain",
            "experience_level",
            "interests",
            "career_goal"
        ]
    ].to_dict(orient="records")

    return jsonify(data)


@app.route("/internships")
def internships():

    data = recommender.internships.to_dict(
        orient="records"
    )

    return jsonify(data)


if __name__ == "__main__":

    print("\n======================================")
    print("         INTERNMATCH AI SYSTEM")
    print("======================================")
    print("AI Internship Recommendation Engine")
    print("Server: http://127.0.0.1:5000")
    print("======================================\n")

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )