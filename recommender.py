import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class InternshipRecommender:

    def __init__(self):

        # Load datasets
        self.students = pd.read_csv("data/students.csv")
        self.internships = pd.read_csv("data/internships.csv")
        self.ratings = pd.read_csv("data/ratings.csv")

        # TF-IDF for content-based recommendation
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        internship_text = (
            self.internships["required_skills"].fillna("") + " " +
            self.internships["domain"].fillna("") + " " +
            self.internships["title"].fillna("")
        )

        self.internship_vectors = self.vectorizer.fit_transform(
            internship_text
        )

    # --------------------------------------------------
    # CONTENT-BASED FILTERING
    # --------------------------------------------------

    def content_based_score(self, student_id):

        student = self.students[
            self.students["student_id"] == student_id
        ]

        if student.empty:
            return np.zeros(len(self.internships))

        student = student.iloc[0]

        profile = (
            str(student["skills"]) + " " +
            str(student["domain"]) + " " +
            str(student["interests"]) + " " +
            str(student["career_goal"])
        )

        student_vector = self.vectorizer.transform(
            [profile]
        )

        scores = cosine_similarity(
            student_vector,
            self.internship_vectors
        )[0]

        return scores

    # --------------------------------------------------
    # COLLABORATIVE FILTERING
    # --------------------------------------------------

    def collaborative_score(self, student_id):

        matrix = self.ratings.pivot_table(
            index="student_id",
            columns="internship_id",
            values="rating",
            fill_value=0
        )

        if student_id not in matrix.index:
            return np.zeros(len(self.internships))

        student_vector = matrix.loc[
            student_id
        ].values.reshape(1, -1)

        # Calculate similarity between students
        similarity = cosine_similarity(
            student_vector,
            matrix.values
        )[0]

        # IMPORTANT:
        # Do not compare the student with themselves
        current_student_index = matrix.index.get_loc(
            student_id
        )

        similarity[current_student_index] = 0

        # Weighted ratings from similar students
        other_scores = np.dot(
            similarity,
            matrix.values
        )

        similarity_sum = similarity.sum()

        if similarity_sum == 0:

            predicted = np.zeros(
                len(matrix.columns)
            )

        else:

            predicted = (
                other_scores /
                similarity_sum
            )

        # Convert matrix scores into internship order
        result = []

        for internship_id in self.internships[
            "internship_id"
        ]:

            if internship_id in matrix.columns:

                index = list(
                    matrix.columns
                ).index(internship_id)

                result.append(
                    predicted[index]
                )

            else:

                result.append(0)

        return np.array(result)

    # --------------------------------------------------
    # HYBRID RECOMMENDATION
    # --------------------------------------------------

    def recommend(self, student_id, top_n=5):

        content_scores = (
            self.content_based_score(student_id)
        )

        collaborative_scores = (
            self.collaborative_score(student_id)
        )

        # Normalize collaborative score
        collaborative_normalized = (
            collaborative_scores / 5
        )

        # Hybrid model
        # 60% Content-Based
        # 40% Collaborative Filtering
        hybrid_scores = (
            0.6 * content_scores +
            0.4 * collaborative_normalized
        )

        results = self.internships.copy()

        results["content_score"] = (
            content_scores
        )

        results["collaborative_score"] = (
            collaborative_scores
        )

        results["match_score"] = (
            hybrid_scores * 100
        )

        # Sort by highest AI match
        results = results.sort_values(
            by="match_score",
            ascending=False
        )

        return results.head(top_n)


# ------------------------------------------------------
# TEST THE RECOMMENDER
# ------------------------------------------------------

if __name__ == "__main__":

    recommender = InternshipRecommender()

    student_id = "S001"

    recommendations = recommender.recommend(
        student_id,
        top_n=5
    )

    print("\n==========================================")
    print("       INTERNMATCH AI RECOMMENDATIONS")
    print("==========================================\n")

    print(
        f"Student ID: {student_id}"
    )

    print("\nTop Internship Matches:\n")

    for index, (_, row) in enumerate(
        recommendations.iterrows(),
        start=1
    ):

        print(
            f"#{index} "
            f"{row['title']} | "
            f"{row['company']} | "
            f"Match: {row['match_score']:.2f}%"
        )

    print("\n==========================================")