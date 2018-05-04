all: random_forest.cpp
	clang-format -i -style=Google ./random_forest.cpp
	g++ random_forest.cpp -std=c++11 -O3 -pthread -Wall -Wextra -Wpedantic -o random_forest 
