# Contributing

1. Create a focused branch.
2. Install `requirements.txt`.
3. Run migrations with `python manage.py migrate`.
4. Run `python manage.py check`.
5. Run `python manage.py test`.
6. Run `python -m compileall -q app aegis_credit src`.
7. If demonstration-model behavior changes, regenerate the deterministic data
   and local-only artifact with `python -m src.generate_demo_data` followed by
   `python -m src.train_model --demo --quick`, then update the model/data cards.
   Signed releases require the separate `--release` controls documented in
   `docs/ML_DEVELOPMENT.md`.
8. If database models change, commit the generated migration and test both a
   fresh database and an upgrade from the previous migration.

Do not commit raw applicant data, local databases, `.env` files, or credentials.
Changes to features, thresholds, costs, or decision language require explicit
model-governance review.
