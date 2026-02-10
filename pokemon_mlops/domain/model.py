from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


class ModelFactory:

    @staticmethod
    def get_logistic_regression():
        return LogisticRegression(
            max_iter=1000
        )

    @staticmethod
    def get_random_forest():
        return RandomForestClassifier(
            random_state=42
        )
