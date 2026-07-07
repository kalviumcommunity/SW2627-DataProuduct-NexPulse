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