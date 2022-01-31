black automl_common test --check || :
isort automl_common test --check || :
mypy automl_common || :
Success: no issues found in 31 source files
flake8 automl_common || :
flake8 test || :
pydocstyle automl_common || :
