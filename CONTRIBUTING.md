# Contributing to SymptomAssist AI

First of all, thank you for your interest in contributing to SymptomAssist AI! Open source relies on developers like you to improve the codebase, fix bugs, and add new features.

This document provides guidelines and steps for contributing to the repository.

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Project Structure](#project-structure)
3. [How to Contribute](#how-to-contribute)
   - [Reporting Bugs](#reporting-bugs)
   - [Suggesting Enhancements](#suggesting-enhancements)
   - [Pull Requests](#pull-requests)
4. [Development Setup](#development-setup)
5. [Coding Guidelines](#coding-guidelines)

## Code of Conduct
By participating in this project, you agree to maintain a welcoming and inclusive environment. Please be respectful and constructive in all discussions and code reviews.

## Project Structure
The repository is organized as follows:
- `app/` - The FastAPI backend, including core reasoning engines (`knowledge_graph.py`, `nlp_extractor.py`, `rag_pipeline.py`) and API routing.
- `data/` - Contains the dataset files `symptom_disease.csv` and `medical_docs.csv` powering the graph traversal.
- `static/` - Frontend assets (HTML, CSS, JS), including the D3.js powered knowledge graph UI.

## How to Contribute

### Reporting Bugs
If you find a bug, please open an Issue on GitHub. Provide as much detail as possible, including:
- Steps to reproduce the bug.
- Expected vs. actual behavior.
- Browser/OS version, or Python terminal traceback.

### Suggesting Enhancements
Have an idea for a new feature? We'd love to hear it! Open an Issue describing the feature, why it would be beneficial, and any potential implementation thoughts you have.

### Pull Requests
1. Fork the repository and create your branch from `main` and make sure that <ins>**you have starred the repository otherwise it your pr may not be considered.**</ins>
2. Ensure you have installed the requirements using `pip install -r requirements.txt`.
3. If you've added code that should be tested, add relevant tests.
4. Update the `README.md` if your code changes the project's setup or interface.
5. Create a Pull Request with a clear description of the changes.

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/SymptomAssist-AI.git
   cd SymptomAssist-AI
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Create a `.env` file in the root directory and add necessary API keys:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```
   *Note: Never commit your `.env` file.*

5. **Run the application:**
   ```bash
   python -m uvicorn app.main:app --reload --port 8000
   ```
   Open `http://localhost:8000` to interact with the UI.

## Coding Guidelines
- **Python:** Follow PEP 8 style guidelines. Include docstrings for all new significant functions/classes.
- **JavaScript:** Ensure modern ES6+ features are used appropriately, keeping performance in mind (especially regarding the D3 graph).
- Keep commit messages clear and concise (e.g., "Add timeout handling for groq API", "Fix CSS layout for graph nodes").

We appreciate your contributions!
