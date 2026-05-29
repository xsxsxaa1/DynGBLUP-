

DYNGBLUP_VERSION = $(shell cat ./VERSION)

# Set this variable to either LNX or MAC
SYS                    = LNX # LNX|MAC (Linux is the default)

DIST_NAME              = dyngblup-$(DYNGBLUP_VERSION)
DEBUG                  =                  # DEBUG mode, set DEBUG=0 for a release
SHOW_COMPILER_WARNINGS =
WITH_OPENBLAS          = 1                # Without OpenBlas uses LAPACK
WITH_LAPACK            =                  # Force linking LAPACK (if OpenBlas lacks it)
WITH_GSLCBLAS          =                  # Force linking gslcblas (if OpenBlas lacks it)
OPENBLAS_LEGACY        =                  # Using older OpenBlas
FORCE_STATIC           =                  # Static linking of libraries
GCC_FLAGS              = -O3 -std=gnu++11 #-Wall #-lgsl -lgslcblas
TRAVIS_CI              =                  # used by TRAVIS for testing
EIGEN_INCLUDE_PATH     = /home/mxye/software/eigen-3.4.0_build/include/eigen3/
OPENBLAS_INCLUDE_PATH  = /home/mxye/software/OpenBLAS-0.3.23/include/
GSL_INCLUDE_PATH = /home/mxye/software/gsl-2.5_build/include/
CXX = g++ 

# --------------------------------------------------------------------
# Edit below this line with caution
# --------------------------------------------------------------------

BIN_DIR  = ./bin
SRC_DIR  = ./src
TEST_SRC_DIR  = ./test/src

ifdef CXX # CXX defined in environment
  CPP = $(CXX)
  CC = $(CXX)
else
  CPP = g++
endif

ifeq ($(CPP), clang++)
  # macOS Homebrew settings (as used on Travis-CI)
  GCC_FLAGS=-O3 -std=c++11 -stdlib=libc++ -isystem/$(OPENBLAS_INCLUDE_PATH) -isystem/$(EIGEN_INCLUDE_PATH) -Wl,-L/opt/OpenBLAS/lib/
endif

ifdef WITH_OPENBLAS
  OPENBLAS=1
  CPPFLAGS += -DOPENBLAS -isystem/$(OPENBLAS_INCLUDE_PATH)
  ifdef OPENBLAS_LEGACY
    # Legacy version (mostly for Travis-CI)
    CPPFLAGS += -DOPENBLAS_LEGACY
  endif
endif

ifdef DEBUG
  CPPFLAGS += -g $(GCC_FLAGS) -fopenmp -isystem/$(GSL_INCLUDE_PATH) -isystem/$(EIGEN_INCLUDE_PATH) -Icontrib/catch-1.9.7 -Isrc
else
  # release mode
  CPPFLAGS += -DNDEBUG $(GCC_FLAGS) -fopenmp -isystem/$(GSL_INCLUDE_PATH) -isystem/$(EIGEN_INCLUDE_PATH) -Icontrib/catch-1.9.7 -Isrc
endif

ifdef SHOW_COMPILER_WARNINGS
  CPPFLAGS += -Wall
endif

ifndef FORCE_STATIC
  
  LIBS = /home/mxye/software/gsl-2.5_build/lib/libgsl.a /home/mxye/software/gsl-2.5_build/lib/libgslcblas.a /home/mxye/workdir/software/lapack-3.12.0/build/lib/libblas.a /home/mxye/workdir/software/lapack-3.12.0/build/lib/liblapack.a -pthread -llapack -lblas -lz -lopenblas
  ifdef WITH_GSLCBLAS
    LIBS += -lgslcblas
  else
    LIBS += -lgfortran -lquadmath
  endif
else
  LIBS = /home/mxye/software/gsl-2.5_build/lib/libgsl.a /home/mxye/software/gsl-2.5_build/lib/libgslcblas.a /home/mxye/workdir/software/lapack-3.12.0/build/lib/libblas.a /home/mxye/workdir/software/lapack-3.12.0/build/lib/liblapack.a -pthread -llapack -lblas -lz -lopenblas
  ifndef TRAVIS_CI # Travis static compile we cheat a little
    CPPFLAGS += -static
  endif
endif

.PHONY: all

OUTPUT = $(BIN_DIR)/dyngblup

# Detailed libary paths, D for dynamic and S for static

ifdef WITH_LAPACK
  LIBS_LNX_D_LAPACK = -llapack
endif
LIBS_MAC_D_LAPACK = -framework Accelerate

ifdef WITH_LAPACK
  ifeq ($(SYS), MAC)
    LIBS += $(LIBS_MAC_D_LAPACK)
  else
    ifndef FORCE_STATIC
      ifdef WITH_OPENBLAS
        LIBS += -lopenblas
      else
        LIBS += $(LIBS_LNX_D_BLAS)
      endif
      LIBS += $(LIBS_LNX_D_LAPACK)
    else
      LIBS += $(LIBS_LNX_S_LAPACK)
    endif
  endif
endif

HDR          = $(wildcard src/*.h) ./src/version.h
SOURCES      = $(wildcard src/*.cpp)

# all
OBJS = $(SOURCES:.cpp=.o)

all: $(OUTPUT)

./src/version.h:
	./scripts/gen_version_info.sh > src/version.h

$(OUTPUT): $(OBJS)
	$(CPP) $(CPPFLAGS) $(OBJS) $(LIBS) -o $(OUTPUT)

$(OBJS): $(HDR)

.SUFFIXES : .cpp .c .o $(SUFFIXES)

./bin/unittests-dyngblup: contrib/catch-1.9.7/catch.hpp $(TEST_SRC_DIR)/unittests-main.o $(TEST_SRC_DIR)/unittests-math.o $(OBJS)
	$(CPP) $(CPPFLAGS) $(TEST_SRC_DIR)/unittests-main.o  $(TEST_SRC_DIR)/unittests-math.o $(filter-out src/main.o, $(OBJS)) $(LIBS) -o ./bin/unittests-dyngblup

unittests: ./bin/unittests-dyngblup
	./bin/unittests-dyngblup

fast-check: all unittests
	rm -vf test/output/*
	cd test && ./dev_test_suite.sh | tee ../dev_test.log
	grep -q 'success rate: 100%' dev_test.log

slow-check: all
	rm -vf test/output/*
	cd test && ./test_suite.sh | tee ../test.log
	grep -q 'success rate: 100%' test.log

lengthy-check: all
	rm -vf test/output/*
	cd test && ./lengthy_test_suite.sh | tee ../lengthy_test.log
	grep -q 'success rate: 100%' lengthy_test.log

check: fast-check slow-check

check-all: check lengthy-check

clean:
	rm $(SRC_DIR)/version.h
	rm -vf $(SRC_DIR)/*.o
	rm -vf $(SRC_DIR)/*~
	rm -vf $(TEST_SRC_DIR)/*.o
	rm -vf $(OUTPUT)
	rm -vf ./bin/unittests-dyngblup

DIST_COMMON = *.md LICENSE VERSION Makefile
DIST_SUBDIRS = src doc example bin

tar: version all
	@echo "Creating $(DIST_NAME)"
	mkdir -p ./$(DIST_NAME)
	cp $(DIST_COMMON) ./$(DIST_NAME)/
	cp -r $(DIST_SUBDIRS) ./$(DIST_NAME)/
	tar cvzf $(DIST_NAME).tar.gz ./$(DIST_NAME)/
	rm -r ./$(DIST_NAME)
