CXX		  := g++
CXXFLAGS := -std=c++17 -Wall -Wextra -Wformat -Wno-unused-parameter
CXXRELEASE := -O2 -s
CXXDEBUG := -g -Og 

BUILD_PATH = objs

EXE = out.exe

INCLUDES = -I.

SOURCES = main.cpp tests/HSL.cpp tests/P3.cpp tests/adobe_RGB_98.cpp tests/0_255_linear.cpp tests/slow_sRGB.cpp tests/runtime_RGB.cpp

DEFS =
LIBS = 
LDFLAGS = 

CPPFLAGS = $(CXXFLAGS) $(CXXRELEASE) $(INCLUDES) $(DEFS)

OBJS := $(addprefix $(BUILD_PATH)/, $(patsubst %.cpp,%.o,$(SOURCES)))

DEPS = $(OBJS:.o=.d)

all: dirs $(EXE)

$(EXE): $(OBJS)
	$(CXX) -o $@ $^ $(LIBS) $(LDFLAGS)

-include $(DEPS)

$(BUILD_PATH)/%.o: %.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -MP -MMD -c -o $@ $<

clean:
	-rm $(TARGET_EXEC) $(OBJS)
	
.PHONY: dirs
dirs:
	@mkdir -p $(dir $(OBJS))
	@mkdir -p $(BUILD_PATH)
