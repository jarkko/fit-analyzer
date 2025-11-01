# Release v0.3.5

**Release Date:** 2025-11-01

## 🎯 Highlights

This release focuses on improving development workflow and code quality automation:

- **Automated Test Badge Updates**: Pre-commit hook now automatically updates the test count badge in README.md
- **Release Safety**: Added CI checks to prevent releasing broken code
- **Code Quality Enforcement**: Added pre-push checklist to ensure all code passes linting and tests before pushing

## 🔧 Changes

### Development Workflow Improvements
- Added pre-commit hook to auto-update test count badge ([46f0e2f](https://github.com/jarkko/fit-analyzer/commit/46f0e2f))
- Removed CI badge update step - now handled locally by git hook ([2572513](https://github.com/jarkko/fit-analyzer/commit/2572513))
- Added CI status check to release workflow ([512173f](https://github.com/jarkko/fit-analyzer/commit/512173f))
- Added critical pre-push checklist documentation ([5c065f0](https://github.com/jarkko/fit-analyzer/commit/5c065f0))

### Code Quality
- Fixed linting issues ([eabee2f](https://github.com/jarkko/fit-analyzer/commit/eabee2f))
- Removed placeholder tests, added proper multisport detection tests ([7f5e21b](https://github.com/jarkko/fit-analyzer/commit/7f5e21b))
- Applied black formatting ([ee9fde1](https://github.com/jarkko/fit-analyzer/commit/ee9fde1))

## 📦 Installation

```bash
pip install fitanalyzer==0.3.5
```

## 🛠️ For Contributors

If you're contributing to this project, make sure to:
1. Install git hooks: `./scripts/install_hooks.sh`
2. Follow the pre-push checklist in `CODE_QUALITY_CHECKLIST.md`

**Full Changelog:** https://github.com/jarkko/fit-analyzer/compare/v0.3.4...v0.3.5
