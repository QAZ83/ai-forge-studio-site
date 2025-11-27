<#
.SYNOPSIS
    AI Forge Studio - Setup and Build Script
    
.DESCRIPTION
    This script helps set up the development environment and build AI Forge Studio.
    
.AUTHOR
    M.3R3
    
.EXAMPLE
    .\setup.ps1 -Check
    .\setup.ps1 -Build
    .\setup.ps1 -Install
#>

param(
    [switch]$Check,
    [switch]$Build,
    [switch]$Install,
    [switch]$Clean,
    [string]$BuildType = "Release"
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Success { Write-Host $args -ForegroundColor Green }
function Write-Warning { Write-Host $args -ForegroundColor Yellow }
function Write-Error { Write-Host $args -ForegroundColor Red }
function Write-Info { Write-Host $args -ForegroundColor Cyan }

# Banner
Write-Host ""
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host "   AI Forge Studio - Setup & Build System" -ForegroundColor White
Write-Host "   Author: M.3R3" -ForegroundColor Gray
Write-Host "============================================================" -ForegroundColor Magenta
Write-Host ""

# Check for prerequisites
function Test-Prerequisites {
    Write-Info "[CHECK] Verifying prerequisites..."
    $allGood = $true

    # CMake
    $cmake = Get-Command cmake -ErrorAction SilentlyContinue
    if ($cmake) {
        $version = (cmake --version | Select-Object -First 1) -replace 'cmake version ', ''
        Write-Success "[OK] CMake $version"
    } else {
        Write-Error "[MISSING] CMake - Install from https://cmake.org"
        $allGood = $false
    }

    # CUDA
    $nvcc = Get-Command nvcc -ErrorAction SilentlyContinue
    if ($nvcc) {
        $version = (nvcc --version | Select-String "release" | ForEach-Object { $_ -replace '.*release ([\d.]+).*', '$1' })
        Write-Success "[OK] CUDA $version"
    } else {
        # Check default path
        $cudaPath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        $cudaVersions = Get-ChildItem $cudaPath -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
        if ($cudaVersions) {
            Write-Warning "[FOUND] CUDA at $($cudaVersions.FullName) - Add to PATH"
            $env:PATH += ";$($cudaVersions.FullName)\bin"
        } else {
            Write-Error "[MISSING] CUDA Toolkit - Install from https://developer.nvidia.com/cuda-downloads"
            $allGood = $false
        }
    }

    # TensorRT
    $tensorrtPaths = @("C:\TensorRT-10.0", "C:\TensorRT", "$env:USERPROFILE\TensorRT")
    $tensorrtFound = $false
    foreach ($path in $tensorrtPaths) {
        if (Test-Path "$path\include\NvInfer.h") {
            Write-Success "[OK] TensorRT at $path"
            $tensorrtFound = $true
            $env:TENSORRT_ROOT = $path
            break
        }
    }
    if (-not $tensorrtFound) {
        Write-Warning "[MISSING] TensorRT - Download from https://developer.nvidia.com/tensorrt"
    }

    # Vulkan
    if ($env:VULKAN_SDK -and (Test-Path $env:VULKAN_SDK)) {
        $version = Split-Path $env:VULKAN_SDK -Leaf
        Write-Success "[OK] Vulkan SDK $version"
    } else {
        # Try to find it
        $vulkanPaths = Get-ChildItem "C:\VulkanSDK" -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1
        if ($vulkanPaths) {
            Write-Success "[OK] Vulkan SDK at $($vulkanPaths.FullName)"
            $env:VULKAN_SDK = $vulkanPaths.FullName
        } else {
            Write-Error "[MISSING] Vulkan SDK - Install from https://vulkan.lunarg.com"
            $allGood = $false
        }
    }

    # Qt
    if ($env:Qt6_DIR -and (Test-Path $env:Qt6_DIR)) {
        Write-Success "[OK] Qt6 at $env:Qt6_DIR"
    } else {
        # Try to find Qt
        $qtPaths = @("C:\Qt", "$env:USERPROFILE\Qt")
        foreach ($basePath in $qtPaths) {
            $qtVersions = Get-ChildItem "$basePath\6.*" -ErrorAction SilentlyContinue | Sort-Object Name -Descending
            foreach ($qtVer in $qtVersions) {
                $msvcPath = Get-ChildItem "$($qtVer.FullName)\msvc*_64" -ErrorAction SilentlyContinue | Select-Object -First 1
                if ($msvcPath) {
                    Write-Success "[OK] Qt6 found at $($msvcPath.FullName)"
                    $env:Qt6_DIR = "$($msvcPath.FullName)\lib\cmake\Qt6"
                    break
                }
            }
            if ($env:Qt6_DIR) { break }
        }
        if (-not $env:Qt6_DIR) {
            Write-Error "[MISSING] Qt6 - Install from https://www.qt.io/download"
            $allGood = $false
        }
    }

    # Visual Studio
    $vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $vsWhere) {
        $vsPath = & $vsWhere -latest -property installationPath
        if ($vsPath) {
            Write-Success "[OK] Visual Studio at $vsPath"
        }
    } else {
        Write-Error "[MISSING] Visual Studio 2022"
        $allGood = $false
    }

    # GPU Info
    Write-Info ""
    Write-Info "[INFO] Detecting NVIDIA GPU..."
    try {
        $gpu = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -like "*NVIDIA*" }
        if ($gpu) {
            Write-Success "[OK] GPU: $($gpu.Name)"
            Write-Info "      Driver: $($gpu.DriverVersion)"
            Write-Info "      VRAM: $([math]::Round($gpu.AdapterRAM / 1GB, 2)) GB"
        } else {
            Write-Warning "[WARNING] No NVIDIA GPU detected"
        }
    } catch {
        Write-Warning "[WARNING] Could not detect GPU"
    }

    Write-Info ""
    if ($allGood) {
        Write-Success "[SUCCESS] All prerequisites met!"
        return $true
    } else {
        Write-Error "[FAILED] Some prerequisites are missing"
        return $false
    }
}

# Build the project
function Build-Project {
    Write-Info "[BUILD] Starting build process..."
    
    $buildDir = "build"
    
    # Create build directory
    if (-not (Test-Path $buildDir)) {
        New-Item -ItemType Directory -Path $buildDir | Out-Null
    }
    
    Push-Location $buildDir
    
    try {
        # Configure with CMake
        Write-Info "[CMAKE] Configuring project..."
        
        $cmakeArgs = @(
            ".."
            "-G", "Visual Studio 17 2022"
            "-A", "x64"
            "-DCMAKE_BUILD_TYPE=$BuildType"
        )
        
        if ($env:TENSORRT_ROOT) {
            $cmakeArgs += "-DTENSORRT_ROOT=$env:TENSORRT_ROOT"
        }
        
        & cmake @cmakeArgs
        
        if ($LASTEXITCODE -ne 0) {
            throw "CMake configuration failed"
        }
        
        # Build
        Write-Info "[BUILD] Compiling..."
        $processors = [Environment]::ProcessorCount
        & cmake --build . --config $BuildType --parallel $processors
        
        if ($LASTEXITCODE -ne 0) {
            throw "Build failed"
        }
        
        # Compile shaders
        Write-Info "[SHADERS] Compiling Vulkan shaders..."
        
        if (-not (Test-Path "shaders")) {
            New-Item -ItemType Directory -Path "shaders" | Out-Null
        }
        
        $glslc = "$env:VULKAN_SDK\Bin\glslc.exe"
        if (Test-Path $glslc) {
            & $glslc "..\shaders\shader.vert" -o "shaders\shader.vert.spv"
            & $glslc "..\shaders\shader.frag" -o "shaders\shader.frag.spv"
            & $glslc "..\shaders\inference.comp" -o "shaders\inference.comp.spv"
            Write-Success "[OK] Shaders compiled"
        } else {
            Write-Warning "[SKIP] glslc not found, shaders not compiled"
        }
        
        Write-Success ""
        Write-Success "[SUCCESS] Build completed!"
        Write-Info "Output: build\$BuildType\AIForgeStudio.exe"
        
    } finally {
        Pop-Location
    }
}

# Clean build
function Clean-Build {
    Write-Info "[CLEAN] Removing build directory..."
    
    if (Test-Path "build") {
        Remove-Item -Recurse -Force "build"
        Write-Success "[OK] Build directory cleaned"
    } else {
        Write-Info "[OK] Build directory does not exist"
    }
}

# Main execution
if ($Check -or (-not $Build -and -not $Install -and -not $Clean)) {
    Test-Prerequisites
}

if ($Clean) {
    Clean-Build
}

if ($Build) {
    if (Test-Prerequisites) {
        Build-Project
    }
}

if ($Install) {
    Write-Info "[INSTALL] Creating installer..."
    
    if (Test-Path "build\$BuildType\AIForgeStudio.exe") {
        # Check for NSIS
        $nsis = Get-Command makensis -ErrorAction SilentlyContinue
        if ($nsis) {
            & makensis installer.nsi
            Write-Success "[OK] Installer created: AIForgeStudio-Setup.exe"
        } else {
            Write-Warning "[SKIP] NSIS not found - install from https://nsis.sourceforge.io"
        }
    } else {
        Write-Error "[ERROR] Build first with -Build flag"
    }
}

Write-Host ""
