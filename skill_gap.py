import pandas as pd


class SkillGapAnalyzer:

    def __init__(self):
        self.students = pd.read_csv("data/students.csv")
        self.internships = pd.read_csv("data/internships.csv")

    def analyze(self, student_id, internship_id):

        student = self.students[
            self.students["student_id"] == student_id
        ]

        internship = self.internships[
            self.internships["internship_id"] == internship_id
        ]

        if student.empty or internship.empty:
            return {
                "matched_skills": [],
                "missing_skills": [],
                "match_percentage": 0
            }

        student = student.iloc[0]
        internship = internship.iloc[0]

        student_skills = {
            skill.strip().lower()
            for skill in str(student["skills"]).split("|")
        }

        required_skills = {
            skill.strip().lower()
            for skill in str(internship["required_skills"]).split("|")
        }

        matched_skills = sorted(
            student_skills.intersection(required_skills)
        )

        missing_skills = sorted(
            required_skills - student_skills
        )

        if required_skills:
            percentage = (
                len(matched_skills) /
                len(required_skills)
            ) * 100
        else:
            percentage = 0

        return {
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "match_percentage": round(percentage, 2)
        }


if __name__ == "__main__":

    analyzer = SkillGapAnalyzer()

    result = analyzer.analyze(
        "S001",
        "I001"
    )

    print("\n===== SKILL GAP ANALYSIS =====")

    print(
        "Skill Match:",
        result["match_percentage"],
        "%"
    )

    print(
        "Matched Skills:",
        ", ".join(result["matched_skills"])
    )

    print(
        "Missing Skills:",
        ", ".join(result["missing_skills"])
    )