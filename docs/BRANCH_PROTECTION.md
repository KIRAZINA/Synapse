# Branch protection for `main`

Apply these settings in GitHub (Settings → Branches → Add rule for `main`):

- Require a pull request before merging.
- Require at least 1 approving review.
- Require status checks to pass before merging:
  - `lint`
  - `test-fast`
  - `build-and-test-windows`
- Require branches to be up to date before merging.
- Do not allow force pushes or deletions.
- Restrict pushes to matching branches (admins included if desired).

Releases are created only from tags matching `v*`; the `release.yml` workflow
does not run on branch pushes.
