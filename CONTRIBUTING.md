# Contributing

Welcome! This project is friendly to people who are new to coding. This page
walks you through setup and the one tool that runs automatically when you
commit. Take it slow — none of these steps can break anything.

## One-time setup

We assume you have [conda](https://docs.conda.io/en/latest/miniconda.html)
installed (Miniconda or Anaconda). From the project folder:

```bash
# 1. Create and activate an environment just for this project
conda create -n darth_vaeder python=3.12
conda activate darth_vaeder

# 2. Install the project AND the contributor tools in one step
pip install -e ".[dev]"

# 3. Turn on the "pre-commit" hooks (the auto-checker, see below)
pre-commit install --hook-type pre-commit --hook-type pre-push
```

That's it. `pip install -e ".[dev]"` installs the project plus `pre-commit`
into your conda environment, so the `pre-commit` command is ready to use.

## What pre-commit does

Every time you run `git commit`, a tool called
[pre-commit](https://pre-commit.com/) automatically:

- **formats your code** so it looks consistent (you don't have to think about
  spacing, quotes, or import order — it just fixes them), and
- **checks for common mistakes** (unused imports, broken syntax, leftover merge
  markers, etc.).

The formatting and most fixes happen **for you**. You don't need to memorize any
style rules.

> **Note:** writing docstrings is encouraged but **not required** — a missing
> docstring will never block your commit.

## The most common surprise

Sometimes your commit will *fail the first time* with a message like
`files were modified by this hook`. **This is normal and not an error.** It
means the formatter cleaned up your files. Just add the changes and commit
again:

```bash
git add -A
git commit -m "your message"   # this second time it will pass
```

## Running the checks yourself

You don't have to wait for a commit:

```bash
pre-commit run              # check the files you've staged
pre-commit run --all-files  # check the whole project
```

## If you ever get truly stuck

You can skip the checks for one commit with `git commit --no-verify`. Use this
sparingly — the automated checks in CI (which run on every pull request) will
still catch issues, so it's best to fix them locally.

## Continuous integration

When you open a pull request, GitHub automatically runs the same checks
(`.github/workflows/lint.yml`). If it shows a red ✗, run
`pre-commit run --all-files` locally, commit the fixes, and push again.
