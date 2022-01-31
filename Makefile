# NOTE: Used on linux, limited support outside of Linux
#
# A simple makefile to help with small tasks related to development of automl_common
# These have been configured to only really run short tasks. Longer form tasks
# are usually completed in github actions.

.PHONY: help install-dev clean clean-build build

help:
	@echo "Makefile automl_common"
	@echo "* install-dev      to install all dev requirements and install pre-commit"
	@echo "* check            to check the source code for issues"
	@echo "* format           to format the code with black and isort"
	@echo "* pre-commit       to run the pre-commit check"
	@echo "* clean            to clean the dist and doc build files"
	@echo "* build            to build a dist"

PYTHON ?= python
CYTHON ?= cython
PYTEST ?= python -m pytest
CTAGS ?= ctags
PIP ?= python -m pip
MAKE ?= make
BLACK ?= black
ISORT ?= isort
PYDOCSTYLE ?= pydocstyle
MYPY ?= mypy
PRECOMMIT ?= pre-commit
FLAKE8 ?= flake8

DIR := ${CURDIR}
DIST := ${CURDIR}/dist
DOCDIR := ${DIR}/doc
INDEX_HTML := file://${DOCDIR}/html/build/index.html

install-dev:
	$(PIP) install -e ".[dev]"
	pre-commit install

check-black:
	$(BLACK) automl_common test --check || :

check-isort:
	$(ISORT) automl_common test --check || :

check-pydocstyle:
	$(PYDOCSTYLE) automl_common || :

check-mypy:
	$(MYPY) automl_common || :
	$(MYPY) test || :

check-flake8:
	$(FLAKE8) automl_common || :
	$(FLAKE8) test || :

# pydocstyle does not have easy ignore rules, instead, we include as they are covered
check: check-black check-isort check-mypy check-flake8 check-pydocstyle

pre-commit:
	$(PRECOMMIT) run --all-files

format-black:
	$(BLACK) automl_common test

format-isort:
	$(ISORT) automl_common test

format: format-black format-isort

clean-build:
	$(PYTHON) setup.py clean
	rm -rf ${DIST}

# Clean up any builds in ./dist as well as doc
clean: clean-build

# Build a distribution in ./dist
build:
	$(PYTHON) setup.py bdist
