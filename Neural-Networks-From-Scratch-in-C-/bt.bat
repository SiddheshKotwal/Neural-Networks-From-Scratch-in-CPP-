@echo off
setlocal enabledelayedexpansion

for %%f in (*.h) do (
    echo #include "common_includes.h" > temp.txt
    type "%%f" >> temp.txt
    move /Y temp.txt "%%f"
)

echo Include statement added to all .h files.
