import numpy as np
import pytest


@pytest.fixture(scope="module")
def ica_data():
    """Create ICA-style data with mixed independent sources."""
    np.random.seed(42)
    n_samples = 100
    t = np.linspace(0, 10, n_samples)
    S = np.column_stack(
        [
            np.sin(2 * t),
            np.sign(np.cos(3 * t)),
            np.random.laplace(0, 1, n_samples),
            np.random.laplace(0, 1, n_samples),
            np.random.laplace(0, 1, n_samples),
            np.random.laplace(0, 1, n_samples),
            np.random.laplace(0, 1, n_samples),
        ]
    )
    d = S.shape[1]
    mixing_matrix = np.random.randn(d, d)
    X = S @ mixing_matrix.T
    X += 0.3 * np.random.randn(*X.shape)
    return X


@pytest.fixture(
    scope="module",
    params=[
        (100, 10),
        (1000, 10),
        (1000, 2),
    ],
    ids=lambda x: f"{x[0]}x{x[1]}",
)
def data(request):
    from sklearn.datasets import make_blobs

    n_samples, n_features = request.param
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=3, random_state=42)
    return X


@pytest.fixture(scope="module")
def other_data(data):
    from sklearn.datasets import make_blobs

    X, _ = make_blobs(n_samples=data.shape[0], n_features=data.shape[1], centers=3, random_state=43)
    return X


@pytest.fixture(scope="module")
def permuted_data(data):
    np.random.seed(42)
    # For single feature, just flip the sign to test permutation
    if data.shape[1] == 1:
        return -data
    else:
        # Create a proper permutation with sign flips
        perm_indices = np.random.permutation(data.shape[1])
        perm = data[:, perm_indices]
        signs = np.sign(np.random.randn(data.shape[1]))
        perm *= signs
        return perm


@pytest.fixture(scope="module")
def linear_mixed_data(data):
    np.random.seed(42)
    return data @ np.random.randn(data.shape[1], data.shape[1])


@pytest.fixture(scope="module")
def affine_mixed_data(data):
    np.random.seed(42)
    return data @ np.random.randn(data.shape[1], data.shape[1]) + np.random.randn(data.shape[1])[None, :]


@pytest.fixture(scope="session")
def consistent_embeddings(request):
    """Generate data with a set of embeddings that are perfectly consistent
    under a specific transformation."""
    n_estimators, n_samples, n_features, indeterminacy_type = request.param
    np.random.seed(42)

    # 1. Create reference embedding
    reference_embedding = np.random.randn(n_samples, n_features)

    embeddings = [reference_embedding]

    # 2. Generate other embeddings based on indeterminacy type
    for _ in range(n_estimators - 1):
        if indeterminacy_type == "identity":
            new_embedding = reference_embedding.copy()
        elif indeterminacy_type == "permutation":
            perm_indices = np.random.permutation(n_features)
            signs = np.sign(np.random.randn(n_features))
            new_embedding = reference_embedding[:, perm_indices] * signs
        elif indeterminacy_type == "linear":
            mixing_matrix = np.random.randn(n_features, n_features)
            while np.linalg.matrix_rank(mixing_matrix) < n_features:
                mixing_matrix = np.random.randn(n_features, n_features)
            new_embedding = reference_embedding @ mixing_matrix.T
        elif indeterminacy_type == "affine":
            mixing_matrix = np.random.randn(n_features, n_features)
            while np.linalg.matrix_rank(mixing_matrix) < n_features:
                mixing_matrix = np.random.randn(n_features, n_features)
            bias = np.random.randn(n_features)
            new_embedding = reference_embedding @ mixing_matrix.T + bias[None, :]
        else:
            raise ValueError(f"Unknown indeterminacy type: {indeterminacy_type}")
        embeddings.append(new_embedding)

    return embeddings, request.param
