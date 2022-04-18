
FOUND_NVCC := $(shell which nvcc 2> /dev/null)

demo: main.o sobel.o grayscale.o
ifdef FOUND_NVCC
	@echo "Using nvcc"
	nvcc -o demo main.o sobel.o grayscale.o -lstdc++ -L /usr/local/lib -lopencv_imgcodecs -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_video
else
	@echo "No nvcc on PATH, using gcc and cpu only code instead."
	gcc -o demo main.o sobel.o grayscale.o -lstdc++ -L /usr/local/lib -lopencv_imgcodecs -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_video
endif

main.o: src/main.cpp
	gcc -o main.o -c src/main.cpp -I /usr/local/include/opencv4

sobel.o: src/sobel.cu src/sobel.cpp
ifdef FOUND_NVCC
	nvcc -o sobel.o -c src/sobel.cu
else
	gcc -o sobel.o -c src/sobel.cpp
endif

grayscale.o: src/grayscale.cu src/grayscale.cpp
ifdef FOUND_NVCC
	nvcc -o grayscale.o -c src/grayscale.cu
else
	gcc -o grayscale.o -c src/grayscale.cpp
endif