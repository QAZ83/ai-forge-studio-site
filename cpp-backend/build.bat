@echo off
REM =============================================================================
REM AI Forge Studio - C++ Backend Build Script (Windows)
REM =============================================================================

setlocal EnableDelayedExpansion

echo.
echo ========================================
echo   AI Forge Studio - C++ Backend Build
echo ========================================
echo.

REM Check for Visual Studio
where cl >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [!] Visual Studio compiler not found in PATH
    echo [!] Please run from "Developer Command Prompt for VS 2022"
    echo     or run: "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat"
    exit /b 1
)

REM Set paths
set "TENSORRT_ROOT=C:\Program Files\NVIDIA\TensorRT"
set "BUILD_DIR=build"
set "BUILD_TYPE=Release"

REM Parse arguments
if "%1"=="Debug" set "BUILD_TYPE=Debug"
if "%1"=="clean" (
    echo Cleaning build directory...
    if exist %BUILD_DIR% rmdir /s /q %BUILD_DIR%
    echo Done!
    exit /b 0
)

REM Create build directory
if not exist %BUILD_DIR% mkdir %BUILD_DIR%

REM Configure with CMake
echo.
echo [1/2] Configuring with CMake...
echo.

cmake -S . -B %BUILD_DIR% ^
    -G "Visual Studio 17 2022" ^
    -A x64 ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DTENSORRT_ROOT="%TENSORRT_ROOT%"

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] CMake configuration failed!
    exit /b 1
)

REM Build
echo.
echo [2/2] Building...
echo.

cmake --build %BUILD_DIR% --config %BUILD_TYPE% --parallel

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Build failed!
    exit /b 1
)

echo.
echo ========================================
echo   Build Successful!
echo ========================================
echo.
echo Executables:
echo   - build\bin\%BUILD_TYPE%\gpu_info.exe
echo   - build\bin\%BUILD_TYPE%\trt_inference.exe
echo.
echo Run with:
echo   .\build\bin\%BUILD_TYPE%\gpu_info.exe
echo   .\build\bin\%BUILD_TYPE%\trt_inference.exe --help
echo.

exit /b 0
