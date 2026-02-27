# dl-performance-based-rul
Code for a data-driven system-level remaining useful life (RUL) prediction framework that preserves an explicit estimation–prediction separation and enables uncertainty-aware prognostics.


## Installation

This project uses `uv` for dependency management.

### 1. Install uv

**macOS (recommended if you use Homebrew):**
```bash
brew install uv
```

**macOS or Linux (if you do not use Homebrew):**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify installation:
```bash
uv --version
```

---

### 2. Clone the repository

```bash
git https://github.com/4belito/deep-performance-rul.git
cd deep-performance-rul
```

---

### 3. Install dependencies

```bash
uv sync
```

---

### 4. Activate the environment

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```powershell
.venv\Scripts\activate
```