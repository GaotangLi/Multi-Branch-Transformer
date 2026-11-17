CXXFLAGS += -O3 -Wall -shared -std=c++17 -fPIC -fdiagnostics-color

# Bind both the pybind11 headers and extension ABI to the interpreter that will
# import the helper.  Callers may override this with `make PYTHON=/path/python`.
PYTHON ?= python3
CPPFLAGS += $(shell "$(PYTHON)" -m pybind11 --includes)

LIBNAME = helpers_cpp
LIBEXT = $(shell "$(PYTHON)" -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

OUT = $(LIBNAME)$(LIBEXT)
SRC = helpers.cpp

default: $(OUT)

$(OUT): $(SRC)
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) $< -o $@
