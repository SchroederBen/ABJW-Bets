# ABJW Bets – GitHub Rules & Workflow Guide

This document outlines the basic GitHub rules and commands for contributing to the ABJW Bets project.
All team members are expected to follow these guidelines to keep the repository clean and organized.

---

## Branching Rules

- `main`
  - Final, stable version of the project
  - Protected branch
  - No direct commits allowed

- `develop`
  - Active development branch
  - Features are merged here first

- `feature/*`
  - Used for individual tasks or features
  - Examples:
    - `feature/data-ingestion`
    - `feature/model-training`
    - `feature/llm-integration`

---

## General Rules

- Do **not** commit directly to `main`
- Always work in `develop` or a `feature/*` branch
- Keep commits small and descriptive
- Pull the latest changes before starting work
- Use Pull Requests for all merges into `develop` or `main`

---

## Git Commands

~~~bash
# 1) Check what branch you are on
git branch

# 2) Switch to develop
git checkout develop

# 3) Pull latest changes (do this before starting work)
git pull origin develop

# 4) Create a feature branch (only when you begin real work)
git checkout -b feature/your-feature-name

# Example feature branch:
# git checkout -b feature/data-ingestion

# 5) Stage changes
git add .

# 6) Commit changes
git commit -m "Short, descriptive commit message"

# 7) Push your branch to GitHub
git push origin feature/your-feature-name

# 8) Merge a feature branch into develop (CLI – use carefully)
# (Prefer a Pull Request on GitHub instead)
git checkout develop
git pull origin develop
git merge feature/your-feature-name
git push origin develop

# 9) Merge develop into main (Final / Milestone)
# (Prefer a Pull Request on GitHub instead)
git checkout main
git pull origin main
git merge develop
git push origin main

# 10) If you hit a merge conflict during a merge:
# - resolve the conflicts in the files
# - then run:
git add .
git commit
~~~

---

## Pull Requests (Required)

1. Push your `feature/*` branch to GitHub
2. Open a Pull Request:
   - Base branch: `develop`
   - Compare branch: `feature/*`
3. Describe what you changed and why
4. At least one teammate reviews and approves
5. Merge after approval

---

## Commit Message Examples

Good:
- `Add initial project structure`
- `Implement logistic regression baseline`
- `Fix feature alignment bug`
- `Add LLM context feature schema`

Bad:
- `stuff`
- `final`
- `pls work`
