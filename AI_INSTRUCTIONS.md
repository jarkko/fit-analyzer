# AI Assistant Instructions for fit-analyzer Project

## Code Quality Standards

### Linting and Style
- **Never disable linting rules without explicit permission**
- Always refactor code to fix linting warnings properly instead of suppressing them. You must not come up with excuses for why some suppression is fine.
- Maintain 10.00/10 pylint score. Nothing below that is fine, not even 9.99.
- Follow PEP 8 and all configured flake8 rules
- Run `make lint` before considering work complete

### Code Organization
- Keep functions focused and single-purpose
- When functions exceed complexity thresholds (>6 returns, >12 branches):
  1. Extract logical chunks into helper functions
  2. Use descriptive names that explain what the helper does
  3. Keep main function as a readable orchestrator
- Avoid deep nesting - prefer early returns and guard clauses

### Code Editing
- Never use placeholder comments like `...existing code...` in code edits
- Always provide complete, exact code in replacements
- Include sufficient context (3-5 lines) to make edits unambiguous

## Testing Standards

### Test-Driven Development
- **Use TDD approach for bug fixes and new features**
- Write tests first, then implement the fix
- All tests must pass before considering work complete
- Current test count: 90 tests - maintain or increase
- Use pytest for all testing
- Run `make test` to verify all tests pass
- We must always keep 100 % code coverage for committed code

### Test Quality
- Tests should be clear, focused, and well-named
- Use fixtures and test data files where appropriate
- Mock external dependencies (APIs, file I/O when appropriate)
- Verify both positive and negative cases

## Development Workflow

### Before Making Changes
1. Understand the existing code structure
2. Write tests that demonstrate the issue or new behavior
3. Run tests to confirm they fail appropriately
4. Implement the fix or feature
5. Verify tests pass
6. Check code quality with linting

### Seeking Permission
- Ask before making architectural changes
- Request approval for non-obvious refactoring
- Get permission before disabling any quality checks
- Clarify requirements when ambiguous

## Project Commands
- `make test` - Run all tests
- `make lint` - Check code quality (must show 10.00/10)
- `make install-dev` - Install all dependencies
