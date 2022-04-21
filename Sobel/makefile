

filter: main.o kernel.o
	nvcc kernel.o main.o -L /usr/local/lib -lopencv_imgcodecs -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_video -o filter


main.o: src/main.cpp
	nvcc -g -c src/main.cpp
kernel.o: src/kernel.cu
	nvcc -g -c src/kernel.cu

clean:
	rm *.o
	rm filter
