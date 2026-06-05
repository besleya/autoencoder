# SPDX-License-Identifier: MIT
CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fopenmp
LDFLAGS  ?= -fopenmp
LDLIBS   ?= -lzstd

OBJS = main.o autoencoder.o data.o
TARGET = main_ae

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(LDFLAGS) $(OBJS) -o $@ $(LDLIBS)

main.o: main.cpp autoencoder.h data.h
	$(CXX) $(CXXFLAGS) -c main.cpp -o $@

autoencoder.o: autoencoder.cpp autoencoder.h
	$(CXX) $(CXXFLAGS) -c autoencoder.cpp -o $@

data.o: data.cpp data.h
	$(CXX) $(CXXFLAGS) -c data.cpp -o $@

clean:
	rm -f $(OBJS) $(TARGET)
