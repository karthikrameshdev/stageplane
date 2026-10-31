# Contributing to Stageplane

Thanks for helping improve Stageplane. Bug reports, documentation, reproducible
measurements, model adapters, and code changes are welcome. Participation follows
the [Code of Conduct](CODE_OF_CONDUCT.md).

## Before starting

Search the [issues](https://github.com/karthikrameshdev/stageplane/issues) and open
pull requests for related work. For a large change, open an issue describing the
problem and proposed behavior so the scope can be discussed first. Report security
vulnerabilities privately using [SECURITY.md](SECURITY.md).

Read the [architecture](docs/architecture.md) and [protocol](docs/protocol.md)
before changing placement, inference, session recovery, or transport behavior.
For nontrivial API work, follow the repository's [AGENTS.md](AGENTS.md) instructions.

## Development setup

Fork the repository, clone your fork, and create a branch:

```sh
git clone https://github.com/YOUR_USERNAME/stageplane.git
cd stageplane
git switch -c fix/describe-the-change
uv sync --extra test --locked
```

Python 3.11 or later and `uv` are required. CI uses Python 3.13 on Linux; local
testing has also used Python 3.12 and 3.13 on Apple Silicon. Docker is needed only
for container workflows and Compose validation.

## Checks and local testing

Run the automated suite and check the diff before opening a pull request:

```sh
uv run --extra test --locked pytest -q
git diff --check
```

The suite uses tiny local checkpoints; a multi-gigabyte model download is not
required. To inspect the dashboard and a worker pool without downloading weights:

```sh
uv run --locked python -m scripts.tiny_checkpoint data/fixture
uv run --locked stageplane distribute data/fixture --store data/fixture-models --layers-per-chunk 1
uv run --locked python -m scripts.dev_cluster --store data/fixture-models --work data/fixture-run
```

Use the development token `local-stageplane-development` in the dashboard at
`http://127.0.0.1:8080`. The fixture has random weights: it checks execution and
recovery, not answer quality. Ctrl-C stops the local cluster.

For Compose changes, validate the configuration with a disposable value:

```sh
STAGEPLANE_TOKEN=configuration-check-only docker compose config --quiet
```

For model downloads, Docker pools, remote bundles, TLS, and measured Qwen runs,
follow [docs/LOCAL_TESTING.md](docs/LOCAL_TESTING.md). Large-model benchmarks are
optional unless needed to establish the change. Report model revision, precision,
kernel, node count, memory/CPU budgets, host resources, prompt/output lengths, and
raw measurements when making performance claims. Distinguish local processes
from separate machines and allocated limits from measured use.

## Making a change

- Keep changes focused and explain the user-visible problem and resulting behavior.
- Add meaningful regression coverage for runtime changes. Documentation-only
  changes should have their links and examples checked.
- Preserve token parity and bounded recovery when changing execution or retries.
- Update documentation and contracts when supported behavior changes.
- Keep credentials, certificates, prompts containing private data, model weights,
  worker caches, and generated logs out of Git. Existing ignored data directories
  are intended for local artifacts.
- Retain license notices for third-party code. Model adapters and benchmarks must
  respect the upstream model's license and access terms.

Use a concise commit title such as `fix: retry timed-out stages on a ready worker`
or `docs: explain published bundles`. The prefixes `feat`, `fix`, `docs`, `test`,
`refactor`, and `chore` make the history easier to scan.

## Pull requests and review

Push your branch to your fork and open a pull request against
`karthikrameshdev/stageplane:main`. Complete the pull-request template, link the
related issue, and state exactly which checks you ran and their results. Explain
any resource limits that prevented a relevant check.

The upstream repository keeps only `main`. Keep development branches in your
fork, including dependency updates. Automatic Dependabot version-update branches
are disabled; submit dependency updates through the same fork-and-review workflow.

Community contributions require one code-owner approval under the project review
policy. Address review feedback and keep applicable checks passing before merge.
The maintainer may squash commits to keep a focused history. Ownership and the
distinction between review policy and GitHub enforcement are documented in
[MAINTAINERS.md](MAINTAINERS.md).

## License

Contributions are made under the project's [Apache License 2.0](LICENSE), unless
explicitly documented otherwise for an existing third-party component. By
submitting a contribution, you confirm that you have the right to contribute it
under those terms. Preserve [NOTICE](NOTICE) and applicable third-party notices.
