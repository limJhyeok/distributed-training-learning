#!/bin/sh
set -x

uv run ruff check . scripts --fix
uv run ruff format . scripts
