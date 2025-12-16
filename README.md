# ITS — Intelligent Tutoring System (Algebra / Linear Algebra)

Simple Streamlit-based ITS for algebra and linear algebra problems (linear equations, vector addition, 2×2 matrix addition) with:
- adaptive question generation,
- topic & difficulty dropdowns,
- step-by-step guides,
- ontology-backed hints (OWL),
- user accounts stored in SQLite,
- attempt logging and basic student statistics.

---

## Quick start

Prerequisites
- Windows (tested)
- Python 3.10+ (Python 3.14 seen in logs — newer is OK)
- Git (optional)
- streamlit, owlready2, other Python packages (see below)

Install dependencies (recommended inside a virtualenv):
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
If you don't have `requirements.txt`, install:
```bash
pip install streamlit owlready2
```

Place files
- Put the ontology file `its_algebra_example.owl` in the project root `c:\AIC\`.
- Ensure the OWL filename matches the path in the app (`OWL_PATH` in `yetunde.py`).

Start the app
```bash
cd c:\AIC
streamlit run yetunde.py
```

Open the browser URL printed by Streamlit (usually http://localhost:8501).

---

## Files and important variables

- `yetunde.py` — main Streamlit application (UI, DB helpers, question generation, reasoner).
- `its_algebra_example.owl` — required OWL ontology (must be present if you want ontology-based hints).
- `its_tutor.db` or `its_streamlit.db` — SQLite DB file (path set by `DB_PATH` in `yetunde.py`). Make sure `DB_PATH` matches your existing DB filename if migrating.

Key constants (edit in `yetunde.py` if needed):
- `DB_PATH` — path to SQLite DB (default in code: `"its_tutor.db"`). Change to `"its_streamlit.db"` if you have that file.
- `OWL_PATH` — path to the OWL file (default points to `its_algebra_example.owl` in same folder).

---

## Features / Usage

- Registration & login — tracks users, hashed passwords (SHA-256).
- Practice page:
  - Dropdown to select Difficulty (easy / medium / hard).
  - Dropdown to select Question Type (scalar equation, vector addition, matrix addition 2×2).
  - Optional step-by-step guide (expandable).
  - Submit answers; app evaluates, logs attempts, updates stats.
  - Ontology reasoner provides contextual hints (falls back to generic hints if OWL or reasoner unavailable).
  - Next Question button advances and clears input.
- Profile page:
  - View statistics and recent attempts
  - Edit full name and password
  - Delete account (danger zone)

---

## Known behaviors & troubleshooting

OWL file
- App expects the OWL file at `OWL_PATH`. If missing, the app will either:
  - stop and show an error (strict mode), or
  - load with generic hint fallbacks (graceful mode) depending on your code branch.
- If you see errors like `FileNotFoundError` from owlready2, verify `its_algebra_example.owl` exists in `c:\AIC` and `OWL_PATH` points to it.

Database / login issues
- If you previously had `its_streamlit.db` but app created `its_tutor.db`, change `DB_PATH` to your existing DB filename and restart.
- Passwords are hashed — ensure you enter the exact same password and email (email lookup is normalized if the code uses normalization).
- Avoid deleting the DB file while the app holds an open connection; close the connection or restart Streamlit before removing the file.

Timezone error
- If you see `can't subtract offset-naive and offset-aware datetimes`, ensure stored `created_at` values include timezone or the code normalizes timezone when reading. The code contains a fix to set `tzinfo=timezone.utc` when missing.

Reset DB (developer only)
- The production UI does not expose reset/debug buttons. If you need to reset the DB manually:
  1. Stop Streamlit.
  2. Ensure no process is holding the DB (close Python/Streamlit).
  3. Delete the DB file: `del c:\AIC\its_tutor.db` (or use File Explorer).
  4. Restart Streamlit — `init_db()` will create the DB and tables.

Git / CLI tips (Windows PowerShell)
- Use `git config` (note the space) not `git.config`.
- PowerShell does not have `touch`. Create `.gitignore` with:
  ```powershell
  New-Item -Path .gitignore -ItemType File -Force
  ```
- To create `.gitignore` contents:
  ```powershell
  Set-Content -Path .gitignore -Value "its_tutor.db`n__pycache__/"
  ```

---

## Development notes

- Reasoner: `owlready2` + Pellet (if installed). The code attempts to run the Pellet reasoner but catches errors if pellet is missing.
- Question generation: implemented in `generate_question(...)` with difficulty and topic overrides. Confirm `st.session_state` logic clears `current_question` when selection changes to avoid stale questions.
- Input clearing: forms use `clear_on_submit=True` and dynamic input key to avoid stale input persisting across reruns.

---

## Adding / modifying content

- To add topics or step-by-step solutions, update `TOPICS` and `STEP_BY_STEP_SOLUTIONS` dicts in `yetunde.py`.
- To change DB filename, edit `DB_PATH` at top of `yetunde.py`.

---

## Contributing

1. Fork and branch.
2. Add features or fixes.
3. Test locally with `streamlit run yetunde.py`.
4. Create PR with description of changes.

---

## License & contact

- License: add your chosen license file (e.g., MIT) in repo root.
- For questions about the app, add contact metadata or open an issue in your project repo.

---

## Example quick-check list

- [ ] `its_algebra_example.owl` present in `c:\AIC`
- [ ] `DB_PATH` points to your existing DB if you have one
- [ ] Python deps installed: `streamlit`, `owlready2`
- [ ] Run: `streamlit run yetunde.py`

---

If you want, I can:
- produce a `requirements.txt` from the environment,
- add a minimal `.gitignore` file,
- insert the README into the repo now.

Which of those would you like me to create next?  
