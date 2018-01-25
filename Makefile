all:
	g++ -O3 -g0 main.cpp -o stock-bot -lboost_filesystem -lboost_system -lboost_thread -lGL -lGLU -lglut -lm;#./stock-bot

debug:
	g++ -O0 -g3 main.cpp -o stock-bot -lboost_filesystem -lboost_system -lboost_thread -lGL -lGLU -lglut -lm;gdb ./stock-bot
