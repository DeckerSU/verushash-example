@echo off

@REM Check for Visual Studio
call set "VSPATH="
if defined VS140COMNTOOLS ( if not defined VSPATH (
  call set "VSPATH=%%VS140COMNTOOLS%%"
) )
if defined VS120COMNTOOLS ( if not defined VSPATH (
 call set "VSPATH=%%VS120COMNTOOLS%%"
) )
if defined VS110COMNTOOLS ( if not defined VSPATH (
 call set "VSPATH=%%VS110COMNTOOLS%%"
) )
if defined VS100COMNTOOLS ( if not defined VSPATH (
 call set "VSPATH=%%VS100COMNTOOLS%%"
) )
if defined VS90COMNTOOLS ( if not defined VSPATH (
 call set "VSPATH=%%VS90COMNTOOLS%%"
) )
if defined VS80COMNTOOLS ( if not defined VSPATH (
 call set "VSPATH=%%VS80COMNTOOLS%%"
) )

@REM check if we already have the tools in the environment
if exist "%VCINSTALLDIR%" (
 goto compile
)

if not defined VSPATH (
 echo You need Microsoft Visual Studio 8, 9, 10, 11, 12, 13 or 15 installed
 pause
 exit
)

@REM set up the environment
if exist "%VSPATH%..\..\vc\vcvarsall.bat" (
 call "%%VSPATH%%..\..\vc\vcvarsall.bat" amd64
 goto compile_x64
)

echo Unable to set up the environment
pause
exit

:compile_x64
del *.o
del  *.obj
rem nvcc -Xptxas -dlcm=ca -Xptxas -dscm=cs -arch=sm_35 -c cuda/haraka.cu -o cuda/kernel.o 
cl /c haraka.c main.c stuff.c /Iinclude /I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\include" /DCURL_STATICLIB 
link /NODEFAULTLIB:LIBCMT /MACHINE:X64 /nologo cuda/kernel.o haraka.obj main.obj stuff.obj kernel32.lib advapi32.lib Ws2_32.lib libcurl.lib cudart.lib jansson.lib legacy_stdio_definitions.lib /OUT:verushash_example.exe /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\lib\x64" /LIBPATH:"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.10586.0\um\x64" /LIBPATH:"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.10586.0\km\x64" /LIBPATH:"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.10586.0\ucrt\x64" /LIBPATH:"C:\Program Files (x86)\Windows Kits\10\Lib\10.0.10586.0\km\x64" /LIBPATH:"./lib"




