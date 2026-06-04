# contributing to orla

Thank you so much for considering contributing to orla! orla is designed to be a community-focused project and runs on individual contributions from amazing people around the world. This document provides guidelines and instructions for contributing to the project.

## code of conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## getting started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/orla.git
   cd orla
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/harvard-cns/orla.git
   ```

## development setup

### prerequisites

- Go 1.26+ — check with `go version`
- [just](https://just.systems) — task runner used for all local pipelines
- [golangci-lint](https://golangci-lint.run/) v2 — for linting
- Docker — required by storage integration tests (testcontainers)

### setup steps

Verify everything is wired up by running the full local CI pipeline:

```bash
just check
```

That runs build, test (with the race detector), lint, and link checks — the same gate CI runs. Individual recipes:

```bash
just build        # compile every package
just test         # tests only
just lint         # golangci-lint v2
just fmt          # go fmt + go mod tidy
just binary       # build bin/orla
just              # list all recipes
```

## making changes

### Workflow

1. Create a branch from `main`:

   ```bash
   git checkout -b <your_github_username>/<descriptive_name>
   ```

2. Write or update tests for your changes.

3. Run `just check` and make sure it's green.

4. Commit your changes with [conventional commit](https://www.conventionalcommits.org/en/v1.0.0/) messages — one sentence each:

   ```bash
   git commit -m "feat: add support for batched feedback ingestion"
   ```

### commit message guidelines

We follow [conventional commits](https://www.conventionalcommits.org/en/v1.0.0/). Examples:

- `feat: add hot reload support for tool backends`
- `fix: resolve race in scheduler queue depth metric`
- `docs: clarify two-persona contract in personas.md`
- `test: add coverage for batch writer drop policy`
- `refactor: simplify config loading logic`
- `chore: bump go-chi to v5.4`

## submitting changes

### pull request process

1. Push your branch to your fork:
   ```bash
   git push origin <your_github_username>/<descriptive_name>
   ```

2. Create a pull request on GitHub.

3. Ensure CI is green before requesting review.

4. Address reviewer feedback.

### pull request checklist

Before submitting, make sure:

- [ ] You've added tests for new functionality
- [ ] `just check` passes locally
- [ ] Documentation under `docs/` is updated if behavior or wire contract changed
- [ ] Commit messages follow the conventional commit format

## testing

Run all tests with the race detector:

```bash
just test
```

For coverage:

```bash
just coverage    # writes coverage.html
```

Storage tests use [testcontainers](https://testcontainers.com/) and require Docker to be running.

## areas for contribution

We welcome contributions in many areas including bug fixes, new features, documentation, tests, tooling, code quality, security patches, and performance optimizations.

## recognition

Contributors will be recognized in:

1. The GitHub contributors list
2. Release notes

Thank you for contributing to orla!
