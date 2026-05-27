@echo off
REM Forwards all arguments to the project venv's Python.
REM Usage from this directory (cmd or PowerShell):
REM     py main.py --scenario dual_objective --policy horizon
REM     py benchmark.py --scenario dual_objective --policy horizon --seeds 5 --workers 5
"%~dp0.venv\Scripts\python.exe" %*
