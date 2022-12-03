grubs:
	nvcc -std=c++17 -O3 -w -rdc=false -o build Main.cu -Iinclude -lcurand -lglfw -lGLEW -lGL