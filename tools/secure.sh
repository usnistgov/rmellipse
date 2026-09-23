echo "Auditing python environment..."
uv run python -m pip_audit
echo "Running bandit for vulnerability testing..."
uv run bandit  -c pyproject.toml -r src/
