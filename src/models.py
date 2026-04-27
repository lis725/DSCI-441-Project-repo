"""Classical machine-learning models for the mask-detection project.

The submitted/default model is intentionally fixed to the best hyperparameters
found during our tuning experiment. This lets a grader retrain the final model
without repeating the expensive GridSearchCV search.

No deep-learning tools are used here. HOG features are extracted outside this
module. The estimator applies scaling, PCA dimensionality reduction, and an
RBF-kernel SVM classifier.
"""
from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import POSITIVE_LABEL, RANDOM_STATE


MODEL_NAME = "milestone"
GRID_MODEL_NAME = "milestone_grid"

# Best setting selected from the final small GridSearchCV experiment.
FINAL_PCA_COMPONENTS = 0.98
FINAL_SVM_C = 10.0
FINAL_SVM_GAMMA = "scale"


def list_available_models() -> list[str]:
    """Return supported model names.

    ``milestone`` is the fixed final model used for submission and deployment.
    ``milestone_grid`` is optional and reproduces the short hyperparameter
    tuning search. It is not required for normal grading or app usage.
    """
    return [MODEL_NAME, GRID_MODEL_NAME]


def _base_svm_pipeline(
    *,
    pca_components=FINAL_PCA_COMPONENTS,
    svm_c=FINAL_SVM_C,
    svm_gamma=FINAL_SVM_GAMMA,
    random_state: int = RANDOM_STATE,
) -> Pipeline:
    """Build a HOG-feature classifier pipeline.

    The HOG feature matrix is provided by ``src.features``. This pipeline only
    handles feature scaling, PCA, and classification.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=pca_components, svd_solver="full")),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=svm_c,
                    gamma=svm_gamma,
                    class_weight="balanced",
                    probability=True,
                    cache_size=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_model(model_name: str = MODEL_NAME, random_state: int = RANDOM_STATE, n_jobs: int = 1):
    """Build the final model or the optional tuning model.

    Parameters
    ----------
    model_name:
        ``milestone`` trains the fixed final model with selected parameters.
        ``milestone_grid`` runs the optional short GridSearchCV tuning search.
    random_state:
        Reproducibility seed.
    n_jobs:
        Parallel jobs for GridSearchCV only. Keep ``1`` on Windows for stable
        memory usage. Ignored by the fixed final model.
    """
    name = model_name.lower().strip()

    if name == MODEL_NAME:
        return _base_svm_pipeline(random_state=random_state)

    if name == GRID_MODEL_NAME:
        # Optional tuning search. This is kept for reproducibility, but it is
        # not the default because it can take hours on a full image dataset.
        base_pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("pca", PCA(svd_solver="full")),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        cache_size=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        )

        param_grid = {
            "pca__n_components": [0.98],
            "clf__C": [3.0, 10.0],
            "clf__gamma": ["scale", 0.0003],
        }

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        f1_with_mask = make_scorer(f1_score, pos_label=POSITIVE_LABEL)

        return GridSearchCV(
            estimator=base_pipeline,
            param_grid=param_grid,
            scoring=f1_with_mask,
            cv=cv,
            n_jobs=n_jobs,
            pre_dispatch="1*n_jobs",
            verbose=2,
            refit=True,
            return_train_score=True,
        )

    raise ValueError(
        f"Unknown model name: {model_name}. Use 'milestone' for the final model "
        "or 'milestone_grid' for optional hyperparameter tuning."
    )
