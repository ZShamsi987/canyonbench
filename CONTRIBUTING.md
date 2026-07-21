# Contributing

Open an issue before changing a public schema or metric. Behavioral changes require tests, documentation, and a changelog entry. Keep raw imagery, telemetry, credentials, model outputs, and annotator-identifying material out of Git.

Development checks:

```bash
python -m pip install -e '.[dev,registration,analysis]'
ruff check .
ruff format --check .
mypy src
pytest --cov
```

Commits must use the contributor's own configured Git identity. The project does not require automated-agent attribution or co-author trailers.

