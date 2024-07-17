#setup
for dir "A Real Dataset"
Install msys2 then by using it install mingw64 which would be having posix thread
add path of msys2/mingw/... to environment variable
install opencv mingw build for c++
add opencv/bin path to environment variables 
use cmake files given in the repository to make dir build and use del CMakeCache.txt
use cmake -DSOURCE_FILE_ARG="your_filename.cpp" -G "MinGW Makefiles" .. 
to compile the .cpp file then run cmake --build . 
then go to specific .exe inside build dir and execute it (MyProject.exe)

for Neural Networks from Scratch in CPP dir use ctrl + shift + B to compile and run the .exe file