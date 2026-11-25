@echo off
setlocal
set "ROOT=%~dp0"
if "%PYTHONPATH%"=="" (
    set "PYTHONPATH=%ROOT%"
) else (
    set "PYTHONPATH=%ROOT%;%PYTHONPATH%"
)
if exist "%ROOT%.venv\Scripts\python.exe" (
    "%ROOT%.venv\Scripts\python.exe" -m paper_qna %*
) else (
    python -m paper_qna %*
)
set "ERR=%ERRORLEVEL%"
endlocal & exit /b %ERR%
