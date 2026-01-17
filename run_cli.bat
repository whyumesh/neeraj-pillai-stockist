@echo off
REM Simple launcher for CLI mode on Windows
cd /d "%~dp0"
python run_cli.py --input EmailAttachments
pause

