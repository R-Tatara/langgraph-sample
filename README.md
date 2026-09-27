![MIT License](https://img.shields.io/badge/license-MIT-blue)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)

# LangGraph Sample

A small chatbot example built with LangGraph and Google Gemini. It classifies a question, answers general-knowledge questions, and returns a fixed fallback for questions requiring private information.

## Prerequisites

- Python 3.12 or later
- uv (Python package manager)
- A Google Gemini API key

## Installation

```bash
git clone https://github.com/R-Tatara/langgraph-sample.git
cd langgraph-sample
uv sync
```

## Usage

Set your API key in the environment before running the sample:

```bash
export GOOGLE_API_KEY="your-api-key"
uv run python main.py
```

## LISENCE

MIT
