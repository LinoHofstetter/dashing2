# Special target that is not associated with a file but with a command to be executed.
.PHONY=clean

#Default C++ & C compiler
CXX?=g++
CC?=gcc

#Cache size default value & flag
CACHE_SIZE?=4194304
CACHE_SIZE_FLAG:=-DD2_CACHE_SIZE=${CACHE_SIZE}
GIT_VERSION?=v2.1.19


# If on M1, use -target arm64-apple-macos11 -mmacosx-version-min=11.0
# Otherwise, use march=native
UNAME_P := $(shell uname -p)
ifeq ($(UNAME_P),arm)
	TARGET_FLAG=-target arm64-apple-macos11 -mmacosx-version-min=11.0
else
	TARGET_FLAG=-march=native
endif

LIB=-lz # -lfmt, library to link against
INC=-IlibBigWig -Ibonsai/include -Ibonsai -Ibonsai/hll -Ibonsai/hll/include -Ibonsai -I. -Isrc -Ifmt/include #includes
OPT+= -O3 \ # additions to the compiler flags
	$(TARGET_FLAG) \ 
	-fopenmp -pipe $(CACHE_SIZE_FLAG)

OPTMV:=$(OPT)
CXXSTD?=-std=c++20
OPT+= $(CXXSTD)
WARNING+=-Wall -Wextra -Wno-unused-function -Wno-char-subscripts -pedantic -Wno-array-bounds # -Wno-shift-count-overflow
EXTRA+=-DNOCURL -DDASHING2_VERSION=\"$(GIT_VERSION)\" -DFMT_HEADER_ONLY
CXXFLAGS+= $(CXXSTD)
CFLAGS+= -std=c11

D2SRC=$(wildcard src/*.cpp)
OFS=$(patsubst %.cpp,%.o,$(wildcard src/*.cpp)) $(patsubst %.c,%.o, $(wildcard src/*.c))
OBJ=$(OFS) src/osfmt.o
OBJLD=$(patsubst %.o,%.ldo,$(OFS)) src/osfmt.o
OBJF=$(patsubst %.o,%.fo,$(OFS)) src/osfmt.o
OBJF64=$(patsubst %.o,%.f64o,$(OFS)) src/osfmt.o
OBJLD64=$(patsubst %.o,%.ld64o,$(OFS)) src/osfmt.o
OBJ64=$(patsubst %.o,%.64o,$(OFS)) src/osfmt.o
OBJDBG=$(patsubst %.o,%.do,$(OFS)) src/osfmt.o
OBJADD=$(patsubst %.o,%.sano,$(OFS)) src/osfmt.o
OBJLTO=$(patsubst %.o,%.lto,$(OFS)) src/osfmt.o
OBJ0=$(patsubst %.o,%.0,$(OFS)) src/osfmt.o
OBJV=$(patsubst %.o,%.vo,$(OFS)) src/osfmt.o
OBJG=$(patsubst %.o,%.gobj,$(OFS)) src/osfmt.o
OBJW=$(patsubst %.o,%.wo,$(OFS)) src/osfmt.o
OBJNLTO=$(patsubst %.o,%.nlto,$(OFS)) src/osfmt.o

# Preprocessing commands for AVX2 and AVX512BW specific object files
D2SRCSTATICAVX2=$(patsubst %.cpp,%.static-avx2.o,$(D2SRC))
D2SRCSTATICAVX512BW=$(patsubst %.cpp,%.static-avx512bw.o,$(D2SRC))

libdashing2: $(OBJ)
	ar rcs $@ $^

%.o: %.cpp # compiles  .cpp files to .o files
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DNDEBUG -O3
%.o: %.c # compiles .c files to .o files
	$(CC) $(INC) $(OPTMV) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DNDEBUG -O3 -std=c11
src/osfmt.o: fmt/src/os.cc
	$(CXX) -I fmt/include $(OPT) $(WARNING) $< -c -o $@ $(EXTRA)

clean:
	rm -f libdashing2 $(OBJ)


##old Makefile

# Special target that is not associated with a file but with a command to be executed.
.PHONY=clean

#Default C++ & C compiler
CXX?=g++
CC?=gcc

#Cache size default value & flag
CACHE_SIZE?=4194304
CACHE_SIZE_FLAG:=-DD2_CACHE_SIZE=${CACHE_SIZE}
GIT_VERSION?=v2.1.19


# If on M1, use -target arm64-apple-macos11 -mmacosx-version-min=11.0
# Otherwise, use march=native
UNAME_P := $(shell uname -p)
ifeq ($(UNAME_P),arm)
	TARGET_FLAG=-target arm64-apple-macos11 -mmacosx-version-min=11.0
else
    TARGET_FLAG=-march=native
endif

LIB=-lz # -lfmt, library to link against
INC=-IlibBigWig -Ibonsai/include -Ibonsai -Ibonsai/hll -Ibonsai/hll/include -Ibonsai -I. -Isrc -Ifmt/include #includes
OPT+= -O3 \ # additions to the compiler flags
    $(TARGET_FLAG) \ 
    -fopenmp -pipe $(CACHE_SIZE_FLAG)

OPTMV:=$(OPT)
CXXSTD?=-std=c++20
OPT+= $(CXXSTD)
WARNING+=-Wall -Wextra -Wno-unused-function -Wno-char-subscripts -pedantic -Wno-array-bounds # -Wno-shift-count-overflow
EXTRA+=-DNOCURL -DDASHING2_VERSION=\"$(GIT_VERSION)\" -DFMT_HEADER_ONLY
CXXFLAGS+= $(CXXSTD)
CFLAGS+= -std=c11

D2SRC=$(wildcard src/*.cpp)
OFS=$(patsubst %.cpp,%.o,$(wildcard src/*.cpp)) $(patsubst %.c,%.o, $(wildcard src/*.c))
OBJ=$(OFS) src/osfmt.o
OBJLD=$(patsubst %.o,%.ldo,$(OFS)) src/osfmt.o
#linking with different types of floating-point precision. I assumed this is not needed in the static library build
OBJF=$(patsubst %.o,%.fo,$(OFS)) src/osfmt.o
OBJF64=$(patsubst %.o,%.f64o,$(OFS)) src/osfmt.o
OBJLD64=$(patsubst %.o,%.ld64o,$(OFS)) src/osfmt.o
OBJ64=$(patsubst %.o,%.64o,$(OFS)) src/osfmt.o
#various debugging options, assumed not needed in the static library build
OBJDBG=$(patsubst %.o,%.do,$(OFS)) src/osfmt.o
OBJADD=$(patsubst %.o,%.sano,$(OFS)) src/osfmt.o
OBJLTO=$(patsubst %.o,%.lto,$(OFS)) src/osfmt.o
# different optimization levels, assumed not needed in the static library build
OBJ0=$(patsubst %.o,%.0,$(OFS)) src/osfmt.o
OBJV=$(patsubst %.o,%.vo,$(OFS)) src/osfmt.o
OBJG=$(patsubst %.o,%.gobj,$(OFS)) src/osfmt.o
OBJW=$(patsubst %.o,%.wo,$(OFS)) src/osfmt.o
OBJNLTO=$(patsubst %.o,%.nlto,$(OFS)) src/osfmt.o

# Preprocessing commands for AVX2 and AVX512BW specific object files, assumed not needed in the static library build
D2SRCSTATICAVX2=$(patsubst %.cpp,%.static-avx2.o,$(D2SRC))
D2SRCSTATICAVX512BW=$(patsubst %.cpp,%.static-avx512bw.o,$(D2SRC))

all: dashing2 dashing2-64 # main build targets -> builds the binaries
unit: readfx readbw readbed 
obh: echo $(OBJ)

all3d: dashing2 dashing2-f dashing2-ld dashing2-64 dashing2-f64 dashing2-ld64
SEDSTR=
ifeq ($(shell uname -s ),Darwin)
    SEDSTR = " '' "
endif

# different build configurations, assumed not needed in the static library build
OBJFS=src/enums.cpp src/counter.cpp src/fastxsketch.cpp src/merge.cpp src/bwsketch.cpp src/bedsketch.cpp src/fastxsketchbyseq.cpp src/bwreduce.cpp
LIBOBJ=$(patsubst %.cpp,%.o,$(OBJFS))  src/osfmt.o
LIB0=$(patsubst %.cpp,%.0,$(OBJFS)) src/osfmt.o
LIBV=$(patsubst %.cpp,%.vo,$(OBJFS)) src/osfmt.o
DLIBOBJ=$(patsubst %.cpp,%.do,$(OBJFS)) src/osfmt.o
GLIBOBJ=$(patsubst %.cpp,%.gobj,$(OBJFS)) src/osfmt.o
FLIBOBJ=$(patsubst %.cpp,%.fo,$(OBJFS)) src/osfmt.o
LONGLIBOBJ=$(patsubst %.cpp,%.64o,$(OBJFS)) src/osfmt.o
LDLIBOBJ=$(patsubst %.cpp,%.ldo,$(OBJFS)) src/osfmt.o

printv:
	echo $(GIT_VERSION)

# compiling & linking binaries for different build configurations/optimization flags, assumed not needed in the static library build
dashing2: dashing2-tmp
	cp $< $@
dashing2-tmp: $(OBJ) libBigWig.a $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJ) -o $@ $(LIB) $(EXTRA) libBigWig.a -DNDEBUG
dashing2-64: $(OBJ64) libBigWig.a
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJ64) -o $@ $(LIB) $(EXTRA) libBigWig.a -DNDEBUG -DLSHIDTYPE="uint64_t"


dashing2-0: $(OBJ0) libBigWig.a $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJ0) -o $@ $(LIB) $(EXTRA) libBigWig.a -UNDEBUG
dashing2-d: $(OBJDBG) libBigWig.a
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJDBG) -o $@ $(LIB) $(EXTRA) libBigWig.a -O0
dashing2-v: $(OBJV) libBigWig.a $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJV) -o $@ $(LIB) $(EXTRA) libBigWig.a
dashing2-d0: $(OBJDBG) libBigWig.a
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJDBG) -o $@ $(LIB) $(EXTRA) libBigWig.a -O0
dashing2-add: $(OBJADD) libBigWig.a
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJADD) -o $@ $(LIB) $(EXTRA) libBigWig.a -fsanitize=address -O1
dashing2-g: $(OBJG) libBigWig.a
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJG) -o $@ $(LIB) $(EXTRA) libBigWig.a -fno-lto -pg
dashing2-ld: $(OBJLD) libBigWig.a
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJLD) -o $@ $(LIB) $(EXTRA) libBigWig.a -DNDEBUG
dashing2-f: $(OBJF) libBigWig.a
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJF) -o $@ $(LIB) $(EXTRA) libBigWig.a -DNDEBUG
dashing2-f64: $(OBJF64) libBigWig.a
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJF64) -o $@ $(LIB) $(EXTRA) libBigWig.a -DNDEBUG -DLSHIDTYPE="uint64_t"
dashing2-ld64: $(OBJLD64) libBigWig.a
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJLD64) -o $@ $(LIB) $(EXTRA) libBigWig.a -DNDEBUG  -DLSHIDTYPE="uint64_t"
dashing2-nolto: $(OBJNLTO) libBigWig.a $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJNLTO) -o $@ $(LIB) $(EXTRA) libBigWig.a -DNDEBUG
read%: test/read%.o $(LIBOBJ)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< $(LIBOBJ) -o $@ $(LIB) $(EXTRA) libBigWig.a
read%-ld: test/read%.ldo $(LDLIBOBJ)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< $(LDLIBOBJ) -o $@ $(LIB) $(EXTRA) libBigWig.a -DDSKETCH_FLOAT_TYPE="long double"
read%-f: test/read%.fo $(FLIBOBJ)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< $(FLIBOBJ) -o $@ $(LIB) $(EXTRA) libBigWig.a -DSKETCH_FLOAT_TYPE="float"
%: test/%.cpp $(LIBOBJ)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< $(LIBOBJ) -o $@ $(LIB) $(EXTRA) libBigWig.a -DSKETCH_FLOAT_TYPE="float"
	# $(wildcard src/*.h)
%.o: %.cpp # compiles  .cpp files to .o files
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DNDEBUG -O3
%.o: %.c # compiles .c files to .o files
	$(CC) $(INC) $(OPTMV) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DNDEBUG -O3 -std=c11
#special rules for different build configurations, assumed not needed in the static library build
%.64o: %.cpp
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DNDEBUG -O3 -DLSHIDTYPE="uint64_t"
%.64o: %.c
	$(CC) $(INC) $(OPTMV) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DNDEBUG -O3 -DLSHIDTYPE="uint64_t" -std=c11
%.0: %.cpp
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -O0 -DNDEBUG
%.0: %.c
	$(CC) $(INC) $(OPTMV) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DNDEBUG -O0 -std=c11
%.vo: %.cpp
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DNDEBUG -O3
%.vo: %.c
	$(CC) $(INC) $(OPTMV) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DNDEBUG -O3 -std=c11
%.lto: %.cpp
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DNDEBUG -flto
%.lto: %.c
	$(CC) $(INC) $(OPTMV) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DNDEBUG -flto -std=c11
%.do: %.cpp $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -O0
%.do: %.c $(wildcard src/*.h)
	$(CC) $(INC) $(OPTMV) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -O0 -std=c11
%.sano: %.cpp $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -fsanitize=address -O1
%.sano: %.c $(wildcard src/*.h)
	$(CC) $(INC) $(OPTMV) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -fsanitize=address -O1
%.gobj: %.cpp $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -pg -fno-lto -DNDEBUG
%.ldo: %.cpp $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DSKETCH_FLOAT_TYPE="long double" -DNDEBUG -flto
%.ldo: %.c $(wildcard src/*.h)
	$(CC) $(INC) $(OPTMV) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DSKETCH_FLOAT_TYPE="long double" -DNDEBUG -flto -std=c11
%.fo: %.cpp $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DSKETCH_FLOAT_TYPE="float" -DNDEBUG
%.fo: %.c $(wildcard src/*.h)
	$(CC) $(INC) $(OPTMV) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DSKETCH_FLOAT_TYPE="float" -DNDEBUG -std=c11
%.ld64o: %.cpp $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DSKETCH_FLOAT_TYPE="long double" -DNDEBUG -flto  -DLSHIDTYPE="uint64_t"
%.f64o: %.cpp $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DSKETCH_FLOAT_TYPE="float" -DNDEBUG  -flto -DLSHIDTYPE="uint64_t"
%.nlto: %.cpp
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DNDEBUG -O3
%.nlto: %.c $(wildcard src/*.h)
	$(CC) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DNDEBUG -O3 -std=c11
%.static-avx2.o: %.cpp
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DNDEBUG -O3 -mno-avx512dq -mno-avx512vl -mno-avx512f -mno-avx512bw -mavx -mavx2 -msse2 -msse4.1 -static-libstdc++ -static-libgcc
%.static-avx512bw.o: %.cpp
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(EXTRA) -DNDEBUG -O3 \
     -mavx512dq -mavx512vl -mavx512bw -mavx512f -mavx -mavx2 -msse2 -msse4.1 \
     -static-libstdc++ -static-libgcc
src/osfmt.o: fmt/src/os.cc
	$(CXX) -I fmt/include $(OPT) $(WARNING) $< -c -o $@ $(EXTRA)


%: src/%
	cp $< $@

mmtest: test/mmtest.cpp src/mmvec.h
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -o $@ $(LIB) $(EXTRA)


BWF=libBigWig/bwRead.o libBigWig/bwStats.o libBigWig/bwValues.o libBigWig/bwWrite.o libBigWig/io.o
bwf:
	echo $(BWF)

libgomp.a:
	ln -sf $(shell $(CXX) --print-file-name=libgomp.a)

EXTRA_STATIC=fmt/src/os.cc  libgomp.a
dashing2_s128: $(D2SRC) $(wildcard src/*.h) $(BWF) $(EXTRA_STATIC)
	$(CXX) $(CXXFLAGS) $(OPT) $(WARNING) $(MACH) $(INC) $(LIB) -mno-avx512dq -mno-avx512vl -mno-avx512f -mno-avx512bw -mno-avx -mno-avx2 -msse2 -msse4.1 -static-libstdc++ -static-libgcc \
    $(EXTRA_STATIC) $(BWF) \
		-DNDEBUG $(D2SRC) -o $@ $(EXTRA) $(LIB) -ldl -lz -DNDEBUG

dashing2_savx: $(D2SRC) $(wildcard src/*.h) $(EXTRA_STATIC) $(BWF)
	$(CXX) $(CXXFLAGS) $(OPT) $(WARNING) $(MACH) $(INC) $(LIB) -mno-avx512dq -mno-avx512vl -mno-avx512f -mno-avx512bw -mavx -mno-avx2 -msse2 -msse4.1 -static-libstdc++ -static-libgcc \
    $(EXTRA_STATIC) $(BWF) \
		-DNDEBUG $(D2SRC) -o $@ $(EXTRA) $(LIB) -ldl -lz -DNDEBUG

dashing2_savx2: $(D2SRC) $(wildcard src/*.h) $(EXTRA_STATIC) $(BWF)
	$(CXX) $(CXXFLAGS) $(OPT) $(WARNING) $(MACH) $(INC) $(LIB) -mno-avx512dq -mno-avx512vl -mno-avx512f -mno-avx512bw -mavx -mavx2 -msse2 -msse4.1 -static-libstdc++ -static-libgcc \
    $(EXTRA_STATIC) $(BWF) \
		-DNDEBUG $(D2SRC) -o $@ $(EXTRA) $(LIB) -ldl -lz -DNDEBUG

dashing2_savx2-nolto: $(D2SRCSTATICAVX2) $(wildcard src/*.h) $(EXTRA_STATIC) $(BWF)
	$(CXX) $(CXXFLAGS) $(OPT) $(WARNING) $(MACH) $(INC) $(LIB) -mno-avx512dq -mno-avx512vl -mno-avx512f -mno-avx512bw -mavx -mavx2 -msse2 -msse4.1 -static-libstdc++ -static-libgcc \
    $(EXTRA_STATIC) $(BWF) \
		-DNDEBUG $(D2SRCSTATICAVX2) -o $@ $(EXTRA) $(LIB) -ldl -lz -DNDEBUG

dashing2_savx512bw-nolto: $(D2SRCSTATICAVX512BW) $(wildcard src/*.h) $(EXTRA_STATIC) $(BWF)
	$(CXX) $(CXXFLAGS) $(OPT) $(WARNING) $(MACH) $(INC) $(LIB) -mno-avx512dq -mno-avx512vl -mno-avx512f -mno-avx512bw -mavx -mavx2 -msse2 -msse4.1 -static-libstdc++ -static-libgcc \
    $(EXTRA_STATIC) $(BWF) \
		-DNDEBUG $(D2SRCSTATICAVX512BW) -o $@ $(EXTRA) $(LIB) -ldl -lz -DNDEBUG

dashing2_s512: $(D2SRC) $(wildcard src/*.h) $(EXTRA_STATIC) $(BWF)
	$(CXX) $(CXXFLAGS) $(OPT) $(WARNING) $(MACH) $(INC) $(LIB) -mno-avx512dq -mno-avx512vl -mno-avx512bw -mavx512f -mavx -mavx2 -msse2 -msse4.1 -static-libstdc++ -static-libgcc \
    $(EXTRA_STATIC) $(BWF) \
		-DNDEBUG $(D2SRC) -o $@ $(EXTRA) $(LIB) -ldl -lz -DNDEBUG

dashing2_s512bw: $(D2SRC) $(wildcard src/*.h) $(EXTRA_STATIC) $(BWF)
	$(CXX) $(CXXFLAGS) $(OPT) $(WARNING) $(MACH) $(INC) $(LIB) -mavx512dq -mavx512vl -mavx512bw -mavx512f -mavx -mavx2 -msse2 -msse4.1 -static-libstdc++ -static-libgcc \
    $(EXTRA_STATIC) $(BWF) -DNDEBUG $(D2SRC) -o $@ $(EXTRA) $(LIB) -ldl -lz -DNDEBUG

dashing2_static: dashing2_s128 dashing2_savx dashing2_savx2 dashing2_s512 dashing2_s512bw
static: dashing2_static

libBigWig/%.o: libBigWig/%.c libBigWig.a
	cd libBigWig && make $(shell basename $@)



libBigWig.a: $(wildcard libBigWig/*.c) $(wildcard libBigWig/*.h)
	cd libBigWig && sed -i $(SEDSTR) 's/HAVE_CURL:/#/' Makefile && $(MAKE) && cp libBigWig.a ..

test: readfx readbw

clean:
	rm -f dashing2 dashing2-ld dashing2-f libBigWig.a $(OBJ) $(OBJLD) $(OBJF) readfx readfx-f readfx-ld readbw readbw readbw-f readbw-ld src/*.0 src/*.do src/*.fo src/*.gobj src/*.ldo src/*.0\
		src/*.vo src/*.sano src/*.ld64o src/*.f64o src/*.64o
