
CXX = g++
CXXFLAGS = -g -Wall -std=c++11
LDLIBS = $(shell pkg-config --cflags --libs opencv)
VPATH = cpp

DeslantImg: main.o
	$(CXX) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $@

ifeq (1,$(GPU))
$(info Compiling for GPU using OpenCL.)
VPATH += cl
LDLIBS += -lOpenCL
CXXFLAGS += -DUSE_GPU

DeslantImg: DeslantImgGPU.o CLWrapper.o

else
$(info Compiling for CPU.)

DeslantImg: DeslantImgCPU.o

endif
