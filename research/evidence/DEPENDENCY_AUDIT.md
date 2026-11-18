# Dependency audit

## Method

Audited the generated, hash-pinned development lock in the Python 3.11.9 virtual environment using pip-audit 2.10.1 and the PyPI advisory database. Command:

```powershell
.\.venv311\Scripts\python.exe -m pip_audit --cache-dir .cache\pip-audit --no-deps --disable-pip --require-hashes -r requirements-dev.lock
```

`--no-deps` is redundant with `--require-hashes`; all packages in this `pip-compile` lock are exact-version pinned and hashed. Network access was enabled for the advisory lookup. Result after lock regeneration: **No known vulnerabilities found**.

## Findings and remediation

The initial lock audit reported 63 advisory records in three packages (MLflow 2.22.5, PyArrow 19.0.1, virtualenv 21.14.5). Advisory records include overlapping aliases for affected releases. Updated project constraints and regenerated both locks; the resolver selected MLflow 3.18.0, PyArrow 23.0.1 and virtualenv 21.15.8. The project test suite was rerun after installing this locked dependency set: 119 passed, 7 skipped.

MLflow required a major-version upgrade to leave the vulnerable 2.x range. Its project use is tracking runs, parameters/metrics/artifacts, model registration and artifact retrieval. The existing test suite and code lint checks passed, but MLflow-backed deployment/storage integration was not exercised. Revalidate those workflows in the target MLflow server before rollout.

This is a point-in-time package advisory check, not a guarantee against unknown vulnerabilities. Regenerate/audit locks regularly. The full environment audit initially resolved the installed dependency graph; use the exact lock audit command above for the recorded result.
