@echo off
cd /d "%~dp0"
python -m pip install -r requirements.txt
if errorlevel 1 (echo No se pudieron instalar dependencias. Verifica Python y tu conexion.& pause & exit /b 1)
python simulador_xor.py
if errorlevel 1 pause
