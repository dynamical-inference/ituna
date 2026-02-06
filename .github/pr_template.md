## Release Checklist

> Full release process documentation: [CONTRIBUTING.md](../CONTRIBUTING.md#release-process)

- [ ] Make sure the [build workflow](https://github.com/dynamical-inference/ituna/actions/workflows/build.yml) passes on the release branch. Also make sure that the [milestone](https://github.com/dynamical-inference/ituna/milestones) related to this release is fully done. Move issues that won't make it into the release to the next milestone, then close the milestone.
- [ ] Head to [`ituna.__init__`](ituna/__init__.py) and make sure that the `__version__` is set correctly.
- [ ] [Create a PR](https://github.com/dynamical-inference/ituna/compare) to `main`.
- [ ] Tag the PR with the `release` [label](https://github.com/dynamical-inference/ituna/labels).
- [ ] The [publish workflow](https://github.com/dynamical-inference/ituna/actions/workflows/publish.yml) will run — if it doesn't start, try removing and re-adding the `release` label (step 4).
- [ ] The `release`-labeled PR will build and push to [TestPyPI](https://test.pypi.org/project/ituna/). Verify the staging version looks correct:
  ```bash
  pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ituna==<version>
  ```
  *Note: If you update the PR, the TestPyPI version will **not** be automatically updated. Remove and re-add the `release` label to trigger a new upload.*
- [ ] If all looks good, tests pass and the PR is reviewed, merge the PR **using rebase merging**.
- [ ] Delete the branch.
- [ ] Checkout the updated `main` branch, create the tag and push it:
  ```bash
  git checkout main && git pull
  git tag v<version>
  git push origin v<version>
  ```
  Use the correct format: `v1.2.3` for stable releases, `v1.2.3a4` or `v1.2.3b4` for alpha/beta.
- [ ] Pushing the tag triggers the [publish workflow](https://github.com/dynamical-inference/ituna/actions/workflows/publish.yml) which builds and uploads the package to [PyPI](https://pypi.org/project/ituna/).
- [ ] Verify the release on PyPI:
  ```bash
  pip install ituna==<version>
  python -c "import ituna; print(ituna.__version__)"
  ```
