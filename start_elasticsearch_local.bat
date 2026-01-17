@echo off
setlocal enabledelayedexpansion

REM =========================================
REM Config chemins
REM =========================================
set "ES_HOME=C:\Users\moura\Downloads\elasticsearch-7.10.2-windows-x86_64\elasticsearch-7.10.2"
set "ES_BIN=%ES_HOME%\bin"
set "ES_JDK=%ES_HOME%\jdk"
set "LOG_DIR=%~dp0logs"
set "ES_LOG=%LOG_DIR%\elasticsearch_stdout.log"

REM =========================================
REM Verifs
REM =========================================
if not exist "%ES_BIN%\elasticsearch.bat" (
  echo ERREUR: elasticsearch.bat introuvable dans: "%ES_BIN%"
  pause
  exit /b 1
)

if not exist "%ES_JDK%\bin\java.exe" (
  echo ERREUR: JDK embarque introuvable: "%ES_JDK%\bin\java.exe"
  pause
  exit /b 1
)

REM =========================================
REM Preparer logs (DESACTIVE)
REM =========================================
REM if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

REM =========================================
REM Forcer ES a utiliser le JDK embarque
REM =========================================
set "JAVA_HOME=%ES_JDK%"
set "PATH=%JAVA_HOME%\bin;%PATH%"

REM Optionnel: limiter memoire (utile si PC modeste)
set "ES_JAVA_OPTS=-Xms1g -Xmx1g"

REM =========================================
REM Demarrage
REM =========================================
echo [INFO] ES_HOME=%ES_HOME%
echo [INFO] JAVA_HOME=%JAVA_HOME%
echo [INFO] Logs: %ES_LOG%
echo.

cd /d "%ES_HOME%"

REM Lancer et garder la fenetre ouverte + log stdout/stderr (DESACTIVE)
REM call "%ES_BIN%\elasticsearch.bat" 1>>"%ES_LOG%" 2>>&1

REM Lancer et garder la fenetre ouverte (stdout/stderr dans la console)
call "%ES_BIN%\elasticsearch.bat"

echo.
echo [INFO] Elasticsearch s'est arrete. Voir: %ES_LOG%
pause
endlocal
