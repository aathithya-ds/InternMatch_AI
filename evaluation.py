import pandas as pd
import numpy as np

from sklearn.metrics import precision_score, recall_score, f1_score

from recommender import InternshipRecommender


def evaluate_recommendations():

    recommender = InternshipRecommender()

    ratings = pd.read_csv("data/ratings.csv")

    y_true = []
    y_pred = []

    for student_id in ratings["student_id"].unique():

        student_ratings = ratings[
            ratings["student_id"] == student_id
        ]

        # Internships rated 4 or 5 are treated as relevant
        actual_relevant = set(
            student_ratings[
                student_ratings["rating"] >= 4
            ]["internship_id"]
        )

        recommendations = recommender.recommend(
            student_id,
            top_n=5
        )

        predicted_relevant = set(
            recommendations["internship_id"]
        )

        for internship_id in recommender.internships[
            "internship_id"
        ]:

            if internship_id in actual_relevant:
                y_true.append(1)
            else:
                y_true.append(0)

            if internship_id in predicted_relevant:
                y_pred.append(1)
            else:
                y_pred.append(0)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    print("\n===== INTERNMATCH AI EVALUATION =====")

    print(f"Precision : {precision:.2f}")
    print(f"Recall    : {recall:.2f}")
    print(f"F1 Score  : {f1:.2f}")

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    evaluate_recommendations()