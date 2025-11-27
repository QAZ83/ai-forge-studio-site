; AI Forge Studio - Windows Installer Script (NSIS)
; Author: M.3R3
;
; Build with: makensis installer.nsi
;

!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "x64.nsh"

;--------------------------------
; General Configuration

Name "AI Forge Studio"
OutFile "AIForgeStudio-Setup.exe"
Unicode True
InstallDir "$PROGRAMFILES64\AI Forge Studio"
InstallDirRegKey HKLM "Software\AIForgeStudio" "InstallPath"
RequestExecutionLevel admin

;--------------------------------
; Version Information

!define PRODUCT_NAME "AI Forge Studio"
!define PRODUCT_VERSION "1.0.0"
!define PRODUCT_PUBLISHER "M.3R3"
!define PRODUCT_WEB_SITE "https://aiforgestudio.com"
!define PRODUCT_UNINST_KEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"

VIProductVersion "1.0.0.0"
VIAddVersionKey "ProductName" "${PRODUCT_NAME}"
VIAddVersionKey "CompanyName" "${PRODUCT_PUBLISHER}"
VIAddVersionKey "LegalCopyright" "© 2024 ${PRODUCT_PUBLISHER}"
VIAddVersionKey "FileDescription" "AI Forge Studio Installer"
VIAddVersionKey "FileVersion" "${PRODUCT_VERSION}"
VIAddVersionKey "ProductVersion" "${PRODUCT_VERSION}"

;--------------------------------
; Interface Settings

!define MUI_ABORTWARNING
!define MUI_ICON "assets\icon.ico"
!define MUI_UNICON "assets\icon.ico"
!define MUI_HEADERIMAGE
!define MUI_HEADERIMAGE_BITMAP "assets\header.bmp"
!define MUI_WELCOMEFINISHPAGE_BITMAP "assets\welcome.bmp"

; Modern UI colors
!define MUI_BGCOLOR "0F172A"
!define MUI_TEXTCOLOR "E2E8F0"

;--------------------------------
; Pages

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "LICENSE"
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_RUN "$INSTDIR\AIForgeStudio.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Launch AI Forge Studio"
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

;--------------------------------
; Languages

!insertmacro MUI_LANGUAGE "English"
!insertmacro MUI_LANGUAGE "Arabic"

;--------------------------------
; Installer Sections

Section "AI Forge Studio (Required)" SecCore
    SectionIn RO
    
    SetOutPath "$INSTDIR"
    
    ; Main executable and libraries
    File "build\Release\AIForgeStudio.exe"
    File "build\Release\*.dll"
    
    ; Qt plugins
    SetOutPath "$INSTDIR\platforms"
    File "build\Release\platforms\*.dll"
    
    SetOutPath "$INSTDIR\styles"
    File "build\Release\styles\*.dll"
    
    SetOutPath "$INSTDIR\imageformats"
    File "build\Release\imageformats\*.dll"
    
    ; Shaders (pre-compiled SPIR-V)
    SetOutPath "$INSTDIR\shaders"
    File "build\shaders\*.spv"
    
    ; Resources
    SetOutPath "$INSTDIR\resources"
    File /r "resources\*.*"
    
    ; Write registry keys
    WriteRegStr HKLM "Software\AIForgeStudio" "InstallPath" "$INSTDIR"
    WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "DisplayName" "${PRODUCT_NAME}"
    WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "UninstallString" "$INSTDIR\Uninstall.exe"
    WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "DisplayIcon" "$INSTDIR\AIForgeStudio.exe"
    WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "Publisher" "${PRODUCT_PUBLISHER}"
    WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "DisplayVersion" "${PRODUCT_VERSION}"
    WriteRegStr HKLM "${PRODUCT_UNINST_KEY}" "URLInfoAbout" "${PRODUCT_WEB_SITE}"
    WriteRegDWORD HKLM "${PRODUCT_UNINST_KEY}" "NoModify" 1
    WriteRegDWORD HKLM "${PRODUCT_UNINST_KEY}" "NoRepair" 1
    
    ; Calculate installed size
    ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
    IntFmt $0 "0x%08X" $0
    WriteRegDWORD HKLM "${PRODUCT_UNINST_KEY}" "EstimatedSize" "$0"
    
    ; Create uninstaller
    WriteUninstaller "$INSTDIR\Uninstall.exe"
SectionEnd

Section "Desktop Shortcut" SecDesktop
    CreateShortCut "$DESKTOP\AI Forge Studio.lnk" "$INSTDIR\AIForgeStudio.exe" "" "$INSTDIR\AIForgeStudio.exe" 0
SectionEnd

Section "Start Menu Shortcuts" SecStartMenu
    CreateDirectory "$SMPROGRAMS\AI Forge Studio"
    CreateShortCut "$SMPROGRAMS\AI Forge Studio\AI Forge Studio.lnk" "$INSTDIR\AIForgeStudio.exe" "" "$INSTDIR\AIForgeStudio.exe" 0
    CreateShortCut "$SMPROGRAMS\AI Forge Studio\Uninstall.lnk" "$INSTDIR\Uninstall.exe" "" "$INSTDIR\Uninstall.exe" 0
SectionEnd

Section "CUDA Runtime" SecCUDA
    SetOutPath "$INSTDIR"
    
    ; CUDA runtime libraries
    File "redist\cudart64_12.dll"
    File "redist\cublas64_12.dll"
    File "redist\cublasLt64_12.dll"
    File "redist\cufft64_11.dll"
    File "redist\curand64_10.dll"
    File "redist\cusolver64_11.dll"
    File "redist\cusparse64_12.dll"
SectionEnd

Section "TensorRT Runtime" SecTensorRT
    SetOutPath "$INSTDIR"
    
    ; TensorRT libraries
    File "redist\nvinfer.dll"
    File "redist\nvinfer_plugin.dll"
    File "redist\nvonnxparser.dll"
    File "redist\nvparsers.dll"
SectionEnd

Section "Visual C++ Runtime" SecVCRedist
    SetOutPath "$TEMP"
    
    ; Check if VC++ Redist is already installed
    ReadRegDWORD $0 HKLM "SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" "Installed"
    ${If} $0 != 1
        File "redist\vc_redist.x64.exe"
        ExecWait '"$TEMP\vc_redist.x64.exe" /install /quiet /norestart'
        Delete "$TEMP\vc_redist.x64.exe"
    ${EndIf}
SectionEnd

;--------------------------------
; Section Descriptions

!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
    !insertmacro MUI_DESCRIPTION_TEXT ${SecCore} "Core application files (required)"
    !insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop} "Create a shortcut on the Desktop"
    !insertmacro MUI_DESCRIPTION_TEXT ${SecStartMenu} "Create shortcuts in the Start Menu"
    !insertmacro MUI_DESCRIPTION_TEXT ${SecCUDA} "NVIDIA CUDA runtime libraries for GPU acceleration"
    !insertmacro MUI_DESCRIPTION_TEXT ${SecTensorRT} "NVIDIA TensorRT libraries for optimized inference"
    !insertmacro MUI_DESCRIPTION_TEXT ${SecVCRedist} "Microsoft Visual C++ Runtime (required if not installed)"
!insertmacro MUI_FUNCTION_DESCRIPTION_END

;--------------------------------
; Uninstaller Section

Section "Uninstall"
    ; Remove files
    Delete "$INSTDIR\AIForgeStudio.exe"
    Delete "$INSTDIR\*.dll"
    Delete "$INSTDIR\Uninstall.exe"
    
    ; Remove directories
    RMDir /r "$INSTDIR\platforms"
    RMDir /r "$INSTDIR\styles"
    RMDir /r "$INSTDIR\imageformats"
    RMDir /r "$INSTDIR\shaders"
    RMDir /r "$INSTDIR\resources"
    RMDir "$INSTDIR"
    
    ; Remove shortcuts
    Delete "$DESKTOP\AI Forge Studio.lnk"
    Delete "$SMPROGRAMS\AI Forge Studio\*.lnk"
    RMDir "$SMPROGRAMS\AI Forge Studio"
    
    ; Remove registry keys
    DeleteRegKey HKLM "Software\AIForgeStudio"
    DeleteRegKey HKLM "${PRODUCT_UNINST_KEY}"
SectionEnd

;--------------------------------
; Functions

Function .onInit
    ; Check for 64-bit Windows
    ${IfNot} ${RunningX64}
        MessageBox MB_OK|MB_ICONSTOP "AI Forge Studio requires 64-bit Windows."
        Abort
    ${EndIf}
    
    ; Check for NVIDIA GPU
    ClearErrors
    ReadRegStr $0 HKLM "SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}\0000" "ProviderName"
    ${If} ${Errors}
        MessageBox MB_YESNO|MB_ICONQUESTION "No NVIDIA GPU detected. AI Forge Studio requires an NVIDIA GPU for GPU acceleration.$\n$\nDo you want to continue anyway?" IDYES +2
        Abort
    ${EndIf}
    
    ; Check for existing installation
    ReadRegStr $0 HKLM "Software\AIForgeStudio" "InstallPath"
    ${If} $0 != ""
        MessageBox MB_YESNO|MB_ICONQUESTION "AI Forge Studio is already installed at:$\n$0$\n$\nDo you want to reinstall?" IDYES +2
        Abort
    ${EndIf}
FunctionEnd

Function .onInstSuccess
    ; Register file associations (optional)
    ; WriteRegStr HKCR ".aiforge" "" "AIForgeStudio.Project"
    ; WriteRegStr HKCR "AIForgeStudio.Project" "" "AI Forge Studio Project"
    ; WriteRegStr HKCR "AIForgeStudio.Project\shell\open\command" "" '"$INSTDIR\AIForgeStudio.exe" "%1"'
FunctionEnd
