"""Model registry with preprocessing kept inside each validation fold."""

from collections import OrderedDict

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def build_models(seed: int = 0) -> OrderedDict[str, object]:
    """Return fresh estimators under a common, explicit configuration."""

    return OrderedDict([
        ("logistic_regression", make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2_000, random_state=seed)
        )),
        ("lda", make_pipeline(StandardScaler(), LinearDiscriminantAnalysis())),
        ("qda", make_pipeline(StandardScaler(), QuadraticDiscriminantAnalysis())),
        ("decision_tree", DecisionTreeClassifier(random_state=seed)),
        ("decision_tree_depth_3", DecisionTreeClassifier(max_depth=3, random_state=seed)),
        ("svm_linear", make_pipeline(StandardScaler(), SVC(kernel="linear", C=1.0))),
        ("svm_rbf", make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale"))),
    ])
