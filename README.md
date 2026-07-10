# NexPulse – Analytics Workspace Setup

## Project Description

This repository contains the development environment and workspace setup for the NexPulse project. It provides a standardized analytics workspace for the team, ensuring every member follows the same project structure, dependencies, and development workflow.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone <repository-url>
cd SW2627-DataProuduct-NexPulse
```

### 2. Create Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the deduplication workflow

Use the built-in demo dataset:

```bash
python scripts/data_workflow.py --demo
```

Or run it against a real CSV file:

```bash
python scripts/data_workflow.py --input data/raw/missing_data.csv --output output/processed.csv
```

The workflow prints before/after dtype changes, deduplication counts, and writes audit files to `output/`.

---

## Project Structure

```
data/
 ├── raw/          → Original datasets
 └── processed/    → Cleaned datasets

notebooks/         → Jupyter notebooks

scripts/           → Python scripts

output/            → Reports, charts and exports
```

---

## Notes

- Do not commit the `venv` folder.
- Do not commit the `.env` file.
- Copy `.env.example` to `.env` and update it with your own values.
- Install all required packages using `requirements.txt`.
- The data workflow supports both the built-in demo dataset and CSV input via CLI arguments.
