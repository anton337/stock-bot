all:
	g++ -O0 -g3 main.cpp -o stock-bot -lboost_filesystem -lboost_system -lGL -lGLU -lglut -lm;./stock-bot
	#| gnuplot -p -e 'plot "/dev/stdin" title "Price" with lines'
