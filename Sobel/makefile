
filter: main.o
	g++ main.o -L /usr/local/lib -lopencv_imgcodecs -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_videoio -lopencv_video -o filter

main.o: src/main.cpp
	g++ -g -c src/main.cpp -o main.o -I /usr/local/include/opencv4/
