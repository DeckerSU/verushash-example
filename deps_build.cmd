@echo off
@REM Check for Visual Studio
call set "VSPATH="
if defined VS140COMNTOOLS ( if not defined VSPATH (
 call set "VSPATH=%%VS140COMNTOOLS%%"
) )

@REM check if we already have the tools in the environment
if exist "%VCINSTALLDIR%" (
 goto compile
)

if not defined VSPATH (
 echo You need Microsoft Visual Studio 15 installed
 pause
 exit
)

@REM set up the environment
if exist "%VSPATH%..\..\vc\vcvarsall.bat" (
 call "%%VSPATH%%..\..\vc\vcvarsall.bat" amd64
 goto compile
)

echo Unable to set up the environment
pause
exit

:compile
rem MSBuild /help
echo.
echo Decker will automatically download and build all needed *.dll and *.lib for you ;)
rem timeout /t 5 /nobreak

set curdir=%~dp0
set curdir=%curdir:~0,-1%

:compile_curl
rem 1. curl

mkdir %curdir%\depends
cd %curdir%\depends

rem https://github.com/alexa/avs-device-sdk/wiki/Optimize-libcurl

git clone https://github.com/curl/curl
mkdir curl-build
cd curl-build
cmake -G "Visual Studio 14 2015 Win64" ../curl -DCURL_STATICLIB:BOOL=ON -DCURL_STATIC_CRT:BOOL=ON -DHTTP_ONLY:BOOL=ON -DENABLE_IPV6:BOOL=OFF -DCMAKE_USE_WINSSL:BOOL -DCMAKE_INSTALL_PREFIX=%curdir%
cmake -G "Visual Studio 14 2015 Win64" ../curl -LA > %curdir%\curl_options.txt
rem cmake --build . --config Release --target libcurl 
rem cmake -P cmake_install.cmake 
cmake --build . --config Release --target INSTALL

rem make -j$(nproc)
rem make install
cd %curdir%

:compile_jansson

rem 2. libjansson (http://www.digip.org/jansson/)

mkdir  %curdir%\depends
cd %curdir%\depends
git clone https://github.com/akheron/jansson
mkdir jansson-build
cd jansson-build
cmake -G "Visual Studio 14 2015 Win64" ../jansson -DCMAKE_INSTALL_PREFIX=%curdir% -DJANSSON_BUILD_SHARED_LIBS:BOOL=OFF -DJANSSON_EXAMPLES:BOOL=OFF -DJANSSON_BUILD_DOCS:BOOL=OFF
rem cmake ../jansson -LA > $curdir/options.txt
rem make -j$(nproc)
rem make install
cmake --build . --config Release --target INSTALL
cd %curdir%

