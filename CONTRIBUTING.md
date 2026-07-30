# Contributing

Open an issue before changing a public schema, gate, camera transform, intervention, or
metric. Behavioral changes require tests, documentation, and a changelog entry. Keep
raw imagery, telemetry, credentials, model outputs, test coordinates, procedural seeds,
and auditor-identifying material out of Git.

Development checks:

```bash
python -m pip install 'uv==0.11.30'
uv sync --frozen --extra trace --extra dev
source .venv/bin/activate
ruff check .
ruff format --check .
mypy src
pytest --cov
```

Regenerate `uv.lock` only when dependencies intentionally change.

Commits must use the contributor's own configured Git identity. The project does not require automated-agent attribution or co-author trailers.
