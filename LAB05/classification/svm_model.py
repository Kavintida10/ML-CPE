from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Kernel params tuned per-kernel. Poly needs a low degree or it gets slow
# and overfits; RBF is the default general-purpose choice.
KERNEL_PARAMS = {
    "linear": {"kernel": "linear", "C": 1},
    "poly":   {"kernel": "poly", "C": 1, "degree": 3, "gamma": "scale"},
    "rbf":    {"kernel": "rbf", "C": 10, "gamma": "scale"},
}


def train_svm(X_train, y_train, kernel="rbf", pca_components=150):
    # Scaler + PCA in one pipeline so test data always gets the same
    # transform. PCA also makes SVM tractable on 10,000 pixel features.
    scaler = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(pca_components, *X_train.shape),
                    whiten=True, random_state=42)),
    ])

    # Fit and transform training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Create SVM model with the params for the requested kernel
    params = KERNEL_PARAMS[kernel]
    model = SVC(cache_size=1000, **params)

    # Train model
    model.fit(X_train_scaled, y_train)

    return model, scaler


def predict_svm(model, scaler, X_test):

    # Apply the same scaling used for training data
    X_test_scaled = scaler.transform(X_test)
    # Predict
    predictions = model.predict(X_test_scaled)

    return predictions