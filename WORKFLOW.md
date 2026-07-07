# GitHub Workflow

## Branching Strategy

- Main branch contains stable and production-ready code.
- New work is created in feature branches.
- Branch naming format:
  - feature/[description]
  - fix/[description]
  - docs/[description]
- Feature branches are deleted after merging.

---

## Commit Message Convention

Format:

[type]: description

Examples:

feat: add data validation
docs: update workflow documentation
fix: resolve validation bug
chore: update dependencies

This keeps commit history clean and easy to understand.

---

## Pull Request Process

- Every feature is submitted using a Pull Request.
- At least one review is required before merging.
- Reviews check:
  - Correctness
  - Readability
  - Data integrity
  - Documentation

---

## GitHub Issues

- Every new feature starts with an Issue.
- Issues include:
  - Title
  - Description
  - Label
  - Assignee
- Issues are closed after the related Pull Request is merged.