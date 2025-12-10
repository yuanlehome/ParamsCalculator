Thanks for your interest in contributing!

How to contribute:
- Use issues for bugs and feature requests
- Keep functions small and composable
- Follow existing naming aligned with real config.json keys
- Avoid adding secrets or credentials
- Update requirements when introducing new dependencies

Run locally:
- Install requirements: `pip install -r requirements.txt`
- Start UI: `streamlit run app.py`

Code style:
- Prefer explicit names: `num_attention_heads`, `moe_intermediate_size`
- Keep imports minimal in `app.py`
- Add docstrings for public functions

Tests:
- Add minimal scripts in `scripts/` for reproductions
- Validate formula totals against enumerated parameters
