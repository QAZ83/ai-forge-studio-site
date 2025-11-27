@echo off
REM ============================================================================
REM AI Forge Studio - Build Script for Windows
REM Author: M.3R3
REM 
REM Requirements:
REM   - Visual Studio 2022 with C++ Desktop Development
REM   - CMake 3.25+
REM   - CUDA Toolkit 12.4+
REM   - TensorRT 10.x
REM   - Vulkan SDK 1.4+
REM   - Qt 6.6+
REM ============================================================================

setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo   AI Forge Studio - Build System
echo   Author: M.3R3
echo ============================================================
echo.

REM Check for required tools
where cmake >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CMake not found. Please install CMake 3.25+
    exit /b 1
)

where nvcc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] CUDA not found in PATH. Checking common locations...
    if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe" (
        set "PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"
        echo [OK] Found CUDA 12.4
    ) else (
        echo [ERROR] CUDA Toolkit not found. Please install CUDA 12.4+
        exit /b 1
    )
)

REM Set up Visual Studio environment
if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    echo [INFO] Setting up Visual Studio 2022 Community environment...
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" (
    echo [INFO] Setting up Visual Studio 2022 Professional environment...
    call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvarsall.bat" x64
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" (
    echo [INFO] Setting up Visual Studio 2022 Enterprise environment...
    call "C:\Program Files\Microsoft Visual Studio\2022\Enterprise\VC\Auxiliary\Build\vcvarsall.bat" x64
) else (
    echo [ERROR] Visual Studio 2022 not found.
    exit /b 1
)

REM Create build directory
set BUILD_DIR=build
if not exist %BUILD_DIR% mkdir %BUILD_DIR%
cd %BUILD_DIR%

REM Configure with CMake
echo.
echo [INFO] Configuring with CMake...
echo.

cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_CUDA_ARCHITECTURES="100" ^
    -DBUILD_SHARED_LIBS=OFF

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] CMake configuration failed!
    cd ..
    exit /b 1
)

REM Build the project
echo.
echo [INFO] Building project...
echo.

cmake --build . --config Release --parallel %NUMBER_OF_PROCESSORS%

if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Build failed!
    cd ..
    exit /b 1
)

REM Compile shaders
echo.
echo [INFO] Compiling shaders...
echo.

if not exist shaders mkdir shaders

set GLSLC=glslc
where glslc >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    REM Try Vulkan SDK location
    if exist "%VULKAN_SDK%\Bin\glslc.exe" (
        set GLSLC=%VULKAN_SDK%\Bin\glslc.exe
    ) else (
        echo [WARNING] glslc not found, skipping shader compilation
        goto :skip_shaders
    )
)

%GLSLC% ..\shaders\shader.vert -o shaders\shader.vert.spv
%GLSLC% ..\shaders\shader.frag -o shaders\shader.frag.spv
%GLSLC% ..\shaders\inference.comp -o shaders\inference.comp.spv

echo [OK] Shaders compiled successfully

:skip_shaders

REM Deploy Qt
echo.
echo [INFO] Deploying Qt dependencies...
echo.

if exist "%Qt6_DIR%\..\..\..\bin\windeployqt.exe" (
    "%Qt6_DIR%\..\..\..\bin\windeployqt.exe" --release --no-translations Release\AIForgeStudio.exe
    echo [OK] Qt deployed successfully
) else (
    echo [WARNING] windeployqt not found, please deploy Qt manually
)

cd ..

echo.
echo ============================================================
echo   Build completed successfully!
echo   Output: build\Release\AIForgeStudio.exe
echo ============================================================
echo.

exit /b 0
