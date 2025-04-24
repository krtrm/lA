# Contributing to LegalEase.app

First off, thank you for considering contributing to LegalEase.app! We welcome contributions from the community to help improve this AI legal assistant for Indian law.

Following these guidelines helps to communicate that you respect the time of the developers managing and developing this open-source project. In return, they should reciprocate that respect in addressing your issue, assessing changes, and helping you finalize your pull requests.

## Code of Conduct

This project and everyone participating in it is governed by a [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to [akshatsingh14372@outlook.com].

## How Can I Contribute?

There are many ways to contribute, from writing code and documentation to reporting bugs and suggesting features.

### Reporting Bugs

If you find a bug, please ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/a3ro-dev/LegalEase/issues). If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/a3ro-dev/LegalEase/issues/new). Be sure to include a **title and clear description**, as much relevant information as possible, and a **code sample or an executable test case** demonstrating the expected behavior that is not occurring.

Provide information like:

*   Your operating system and version.
*   Python version.
*   Any relevant environment variables (excluding sensitive keys!).
*   Steps to reproduce the bug.
*   Expected behavior.
*   Actual behavior.
*   Screenshots (if applicable).

### Suggesting Enhancements

If you have an idea for a new feature or an improvement to an existing one, please open an issue on GitHub. Clearly describe the proposed enhancement, why it would be beneficial, and any potential implementation ideas.

### Code Contributions

1.  **Fork the Repository:** Start by forking the main repository to your own GitHub account.
2.  **Clone Your Fork:** Clone your forked repository to your local machine:
    ```bash
    git clone https://github.com/YOUR_USERNAME/LegalEase.git
    cd LegalEase
    ```
3.  **Set Up Development Environment:** Follow the [Setup Instructions in the README.md](README.md#setup-instructions) to install prerequisites and dependencies within a virtual environment.
4.  **Create a Branch:** Create a new branch for your feature or bug fix. Use a descriptive name (e.g., `feature/add-new-tool`, `fix/citation-parser-bug`).
    ```bash
    git checkout -b feature/your-feature-name
    ```
5.  **Make Changes:** Write your code! Ensure you follow the project's coding style (e.g., PEP 8 for Python) and add comments where necessary.
6.  **Add Tests (If Applicable):** If you're adding a new feature or fixing a bug, please add relevant tests to ensure correctness and prevent regressions.
7.  **Ensure Tests Pass:** Run the existing test suite (if available) to make sure your changes haven't broken anything.
8.  **Commit Changes:** Commit your changes with a clear and concise commit message.
    ```bash
    git add .
    git commit -m "feat: Add feature X" # Or "fix: Resolve issue Y"
    ```
    *(Consider using Conventional Commits format: https://www.conventionalcommits.org/)*
9.  **Push to Your Fork:** Push your branch to your fork on GitHub.
    ```bash
    git push origin feature/your-feature-name
    ```
10. **Open a Pull Request (PR):** Go to the original repository on GitHub and open a Pull Request from your branch to the main repository's `main` (or `master`) branch.
    *   Provide a clear title and description for your PR, explaining the changes and referencing any related issues (e.g., "Closes #123").
    *   Be prepared to discuss your changes and make further modifications if requested by the maintainers.

### Documentation

Improvements to documentation (like the README, code comments, or usage examples) are always welcome. Follow the same PR process as for code contributions.

## Coding Style

*   **Python:** Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) guidelines. Use a linter like Flake8 or Black to help enforce consistency.
*   **Comments:** Write clear and concise comments to explain complex logic or non-obvious code sections.
*   **Type Hinting:** Use Python type hints for function signatures and variables where appropriate.

## Questions?

If you have questions about contributing, feel free to open an issue and ask!

Thank you for your interest in contributing to LegalEase.app!