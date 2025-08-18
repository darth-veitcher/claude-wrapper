# GitHub Actions Workflows

This directory contains the CI/CD workflows for the Claude Wrapper project.

## Workflows

### 1. CI (`ci.yml`)
**Trigger:** Push/PR to `main` or `develop` branches

**Features:**
- Tests across Python 3.10, 3.11, 3.12
- Code quality checks (ruff, mypy)
- Security scanning (pip-audit, bandit)
- Code coverage reporting
- Artifact upload for security reports

### 2. Release to PyPI (`release.yml`)
**Trigger:** Push to `main` branch (code changes only)

**Features:**
- Version change detection (only releases new versions)
- Automated PyPI publishing using trusted publishing
- GitHub release creation with tags
- Full test suite before release

**Setup Required:**
1. Configure trusted publishing on PyPI:
   - Go to PyPI project settings
   - Add GitHub as trusted publisher
   - Repository: `your-username/claude-wrapper`
   - Workflow: `release.yml`
   - Environment: `pypi`

### 3. Test Release (`test-release.yml`)
**Trigger:** Push to `develop` branch (code changes only)

**Features:**
- Automatic development version generation
- TestPyPI publishing for testing
- Installation verification from TestPyPI
- Commit comments with installation instructions

**Setup Required:**
1. Configure trusted publishing on TestPyPI:
   - Go to TestPyPI project settings
   - Add GitHub as trusted publisher
   - Repository: `your-username/claude-wrapper`
   - Workflow: `test-release.yml`
   - Environment: `testpypi`

## GitHub Environment Setup

### For PyPI (Production):
1. Go to Settings → Environments → New environment
2. Name: `pypi`
3. Add deployment protection rules (optional)
4. Configure trusted publishing (see above)

### For TestPyPI (Development):
1. Go to Settings → Environments → New environment
2. Name: `testpypi`
3. Configure trusted publishing (see above)

## Security Features

- **Trusted Publishing**: No API tokens needed, uses OpenID Connect
- **Version Detection**: Prevents duplicate releases
- **Security Scanning**: Automated vulnerability detection
- **Environment Protection**: Can require manual approval for releases

## Development Workflow

1. **Feature Development:**
   - Create feature branch from `develop`
   - Make changes
   - Push to feature branch (triggers CI)
   - Create PR to `develop`

2. **Development Testing:**
   - Merge to `develop` (triggers TestPyPI release)
   - Test installation from TestPyPI
   - Verify functionality

3. **Production Release:**
   - Create PR from `develop` to `main`
   - Merge to `main` (triggers PyPI release)
   - Version must be bumped in `pyproject.toml`

## Version Management

- **Main Branch**: Uses version from `pyproject.toml` directly
- **Develop Branch**: Generates dev versions like `0.1.0.dev20241201123456+abcd1234`
- Only new versions are published (automatic duplicate detection)

## Troubleshooting

### Release Not Triggered
- Check if code files were changed (workflows only run on code changes)
- Verify version was bumped in `pyproject.toml`
- Check workflow logs for errors

### Authentication Errors
- Verify trusted publishing is configured correctly
- Check repository name matches exactly
- Ensure environment names match (`pypi`, `testpypi`)

### Test Failures
- All tests must pass before any release
- Check CI workflow for specific failure details
- Security issues will block releases
