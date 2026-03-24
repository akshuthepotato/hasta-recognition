# AGENTS.md

This repository uses `uv` for Python environment, dependency, and command management.

## Package management

- Use `uv` for everything.
- Add runtime dependencies with `uv add <package>`.
- Add development dependencies with `uv add --dev <package>`.
- Remove dependencies with `uv remove <package>` or `uv remove --dev <package>`.
- Run Python entrypoints and tools with `uv run ...`.

## Development workflow

- Put all developer tooling in the `dev` dependency group.
- Prefer commands such as `uv run pytest`, `uv run ruff check`, `uv run ruff format`, and `uv run python ...`.
- If a new CLI tool is needed for local development, install it with `uv add --dev <tool>`.

## pyproject.toml policy

- Never edit `pyproject.toml` by hand for dependency changes.
- Use `uv add`, `uv add --dev`, and `uv remove` so the manifest and lockfile stay in sync.
- If metadata in `pyproject.toml` ever needs to change for a non-dependency reason, make that change deliberately and separately from dependency management.
