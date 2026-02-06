import packaging.version
import sklearn
import sklearn.decomposition

import ituna.estimator


def test_sklearn_compatibility():
    print(f"Testing with sklearn version: {sklearn.__version__}")
    estimator = sklearn.decomposition.FastICA(n_components=7, max_iter=2000)
    ensemble = ituna.estimator.ConsistencyEnsemble(estimator=estimator)

    # Test tag availability based on sklearn version
    sklearn_version = packaging.version.parse(sklearn.__version__)
    is_new_version = sklearn_version >= packaging.version.parse("1.6.dev")

    if is_new_version:
        # Test new __sklearn_tags__ method (sklearn >= 1.6)
        print("Testing __sklearn_tags__ method (sklearn >= 1.6)")
        assert hasattr(ensemble, "__sklearn_tags__"), "ConsistencyEnsemble should have __sklearn_tags__ method"

        tags = ensemble.__sklearn_tags__()
        assert tags is not None, "Tags should not be None"

        # Test critical tags
        assert hasattr(tags, "non_deterministic"), "Tags should have non_deterministic attribute"
        assert not tags.non_deterministic, "ConsistencyEnsemble should inherit deterministic nature from FastICA"

        assert hasattr(tags, "requires_fit"), "Tags should have requires_fit attribute"
        assert tags.requires_fit, "ConsistencyEnsemble should require fitting"

        assert hasattr(tags, "no_validation"), "Tags should have no_validation attribute"
        assert not tags.no_validation, "ConsistencyEnsemble should validate inputs"

        # Test input tags inheritance
        assert hasattr(tags, "input_tags"), "Tags should have input_tags"
        assert hasattr(tags.input_tags, "allow_nan"), "Input tags should have allow_nan"
        assert hasattr(tags.input_tags, "sparse"), "Input tags should have sparse"

        # Test estimator type inheritance
        assert hasattr(tags, "estimator_type"), "Tags should have estimator_type"

        print("✓ __sklearn_tags__ method works correctly")

        # _more_tags should not be available in new versions
        assert not hasattr(ensemble, "_more_tags") or not callable(getattr(ensemble, "_more_tags", None)), (
            "_more_tags should not be available in sklearn >= 1.6"
        )

    else:
        # Test legacy _more_tags method (sklearn < 1.6)
        print("Testing _more_tags method (sklearn < 1.6)")
        assert hasattr(ensemble, "_more_tags"), "ConsistencyEnsemble should have _more_tags method"

        tags = ensemble._more_tags()
        assert isinstance(tags, dict), "Tags should be a dictionary"

        # Test that _more_tags follows sklearn pattern: returns only overridden tags
        # FastICA's _more_tags returns {'preserves_dtype': [...]}, so that's what we should get
        assert "preserves_dtype" in tags, "Should inherit FastICA's preserves_dtype override"

        # These tags are NOT in FastICA's _more_tags, so they shouldn't be in ours either
        # (sklearn will use defaults for these)
        assert "non_deterministic" not in tags, "_more_tags should only contain overridden tags"
        assert "requires_fit" not in tags, "_more_tags should only contain overridden tags"
        assert "no_validation" not in tags, "_more_tags should only contain overridden tags"

        print("✓ _more_tags method works correctly")

        # __sklearn_tags__ should not be the primary method in old versions
        if hasattr(ensemble, "__sklearn_tags__"):
            print("Note: __sklearn_tags__ method also exists but _more_tags takes precedence")

    print(f"✓ Tag compatibility test passed for sklearn {sklearn.__version__}")
