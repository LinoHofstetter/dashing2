.PHONY=clean

CXX?=g++

CACHE_SIZE?=4194304
CACHE_SIZE_FLAG:=-DD2_CACHE_SIZE=${CACHE_SIZE}

LIB=-lz
INC=-IlibBigWig -Ibonsai/include -Ibonsai -Ibonsai/hll -Ibonsai/hll/include -Ibonsai -I. -Isrc
OPT+= -O3 -march=native -fopenmp -pipe $(CACHE_SIZE_FLAG)
OPTMV:=$(OPT)
OPT+= -std=c++17
WARNING+=-Wall -Wextra -Wno-unused-function -Wno-char-subscripts -pedantic # -Wno-shift-count-overflow
EXTRA+=-DNOCURL
CXXFLAGS+= -std=c++17
CFLAGS+= -std=c11

D2SRC=$(wildcard src/*.cpp)
OFS=$(patsubst %.cpp,%.o,$(wildcard src/*.cpp)) $(patsubst %.c,%.o, $(wildcard src/*.c))
OBJ=$(OFS)
OBJLD=$(patsubst %.o,%.ldo,$(OFS))
OBJF=$(patsubst %.o,%.fo,$(OFS))
OBJF64=$(patsubst %.o,%.f64o,$(OFS))
OBJLD64=$(patsubst %.o,%.ld64o,$(OFS))
OBJ64=$(patsubst %.o,%.64o,$(OFS))
OBJDBG=$(patsubst %.o,%.do,$(OFS))
OBJADD=$(patsubst %.o,%.sano,$(OFS))
OBJLTO=$(patsubst %.o,%.lto,$(OFS))
OBJ0=$(patsubst %.o,%.0,$(OFS))
OBJV=$(patsubst %.o,%.vo,$(OFS))
OBJG=$(patsubst %.o,%.gobj,$(OFS))
OBJW=$(patsubst %.o,%.wo,$(OFS))

all: dashing2 dashing2-64
unit: readfx readbw readbed
obh: echo $(OBJ)

all3d: dashing2 dashing2-f dashing2-ld dashing2-64 dashing2-f64 dashing2-ld64
SEDSTR=
ifeq ($(shell uname -s ),Darwin)
    SEDSTR = " '' "
endif

OBJFS=src/enums.cpp src/counter.cpp src/fastxsketch.cpp src/merge.cpp src/bwsketch.cpp src/bedsketch.cpp src/fastxsketchbyseq.cpp src/bwreduce.cpp
LIBOBJ=$(patsubst %.cpp,%.o,$(OBJFS))
LIB0=$(patsubst %.cpp,%.0,$(OBJFS))
LIBV=$(patsubst %.cpp,%.vo,$(OBJFS))
DLIBOBJ=$(patsubst %.cpp,%.do,$(OBJFS))
GLIBOBJ=$(patsubst %.cpp,%.gobj,$(OBJFS))
FLIBOBJ=$(patsubst %.cpp,%.fo,$(OBJFS))
LONGLIBOBJ=$(patsubst %.cpp,%.64o,$(OBJFS))
LDLIBOBJ=$(patsubst %.cpp,%.ldo,$(OBJFS))

dashing2: dashing2-tmp
	cp $< $@
dashing2-tmp: $(OBJ) libBigWig.a $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJ) -o $@ $(LIB) $(EXTRA) libBigWig.a -DNDEBUG -flto
dashing2-64: $(OBJ64) libBigWig.a
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJ64) -o $@ $(LIB) $(EXTRA) libBigWig.a -DNDEBUG -DLSHIDTYPE="uint64_t"


dashing2-0: $(OBJ0) libBigWig.a $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJ0) -o $@ $(LIB) $(EXTRA) libBigWig.a
dashing2-d: $(OBJDBG) libBigWig.a
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJDBG) -o $@ $(LIB) $(EXTRA) libBigWig.a -O0
dashing2-v: $(OBJV) libBigWig.a $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJV) -o $@ $(LIB) $(EXTRA) libBigWig.a
dashing2-d0: $(OBJDBG) libBigWig.a
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJDBG) -o $@ $(LIB) $(EXTRA) libBigWig.a -O0
dashing2-add: $(OBJADD) libBigWig.a
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJADD) -o $@ $(LIB) $(EXTRA) libBigWig.a -fsanitize=thread
dashing2-g: $(OBJG) libBigWig.a
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJG) -o $@ $(LIB) $(EXTRA) libBigWig.a -fno-lto -pg
dashing2-ld: $(OBJLD) libBigWig.a
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJLD) -o $@ $(LIB) $(EXTRA) libBigWig.a -DNDEBUG -flto
dashing2-f: $(OBJF) libBigWig.a
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJF) -o $@ $(LIB) $(EXTRA) libBigWig.a -DNDEBUG -flto
dashing2-f64: $(OBJF64) libBigWig.a
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJF64) -o $@ $(LIB) $(EXTRA) libBigWig.a -DNDEBUG -flto  -DLSHIDTYPE="uint64_t"
dashing2-ld64: $(OBJLD64) libBigWig.a
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJLD64) -o $@ $(LIB) $(EXTRA) libBigWig.a -DNDEBUG -flto  -DLSHIDTYPE="uint64_t"
dashing2-w: $(OBJW) libBigWig.a $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $(OBJW) -o $@ $(LIB) $(EXTRA) libBigWig.a -DNDEBUG -flto -DWANGHASH_EXTRA=1
read%: test/read%.o $(LIBOBJ)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< $(LIBOBJ) -o $@ $(LIB) $(EXTRA) libBigWig.a
read%-ld: test/read%.ldo $(LDLIBOBJ)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< $(LDLIBOBJ) -o $@ $(LIB) $(EXTRA) libBigWig.a -DDSKETCH_FLOAT_TYPE="long double"
read%-f: test/read%.fo $(FLIBOBJ)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< $(FLIBOBJ) -o $@ $(LIB) $(EXTRA) libBigWig.a -DSKETCH_FLOAT_TYPE="float"
%: test/%.cpp $(LIBOBJ)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< $(LIBOBJ) -o $@ $(LIB) $(EXTRA) libBigWig.a -DSKETCH_FLOAT_TYPE="float"
	# $(wildcard src/*.h)
%.o: %.cpp
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -DNDEBUG -flto -O3
%.wo: %.cpp
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -DNDEBUG -flto -O3 -DWANGHASH_EXTRA=1
%.o: %.c
	$(CC) $(INC) $(OPTMV) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -DNDEBUG -flto -O3 -std=c11
%.64o: %.cpp
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -DNDEBUG -flto -O3 -DLSHIDTYPE="uint64_t"
%.64o: %.c
	$(CC) $(INC) $(OPTMV) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -DNDEBUG -flto -O3 -DLSHIDTYPE="uint64_t" -std=c11
%.0: %.cpp
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -DNDEBUG -O0
%.0: %.c
	$(CC) $(INC) $(OPTMV) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -DNDEBUG -O0 -std=c11
%.vo: %.cpp
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -DNDEBUG -O3
%.vo: %.c
	$(CC) $(INC) $(OPTMV) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -DNDEBUG -O3 -std=c11
%.lto: %.cpp
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -DNDEBUG -flto
%.lto: %.c
	$(CC) $(INC) $(OPTMV) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -DNDEBUG -flto -std=c11
%.do: %.cpp $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -O0
%.do: %.c $(wildcard src/*.h)
	$(CC) $(INC) $(OPTMV) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -O0 -std=c11
%.sano: %.cpp $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -flto -fsanitize=address
%.sano: %.c $(wildcard src/*.h)
	$(CC) $(INC) $(OPTMV) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -flto -fsanitize=address
%.gobj: %.cpp $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -pg -fno-lto -DNDEBUG
%.ldo: %.cpp $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -DSKETCH_FLOAT_TYPE="long double" -DNDEBUG -flto
%.ldo: %.c $(wildcard src/*.h)
	$(CC) $(INC) $(OPTMV) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -DSKETCH_FLOAT_TYPE="long double" -DNDEBUG -flto -std=c11
%.fo: %.cpp $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -DSKETCH_FLOAT_TYPE="float" -DNDEBUG
%.fo: %.c $(wildcard src/*.h)
	$(CC) $(INC) $(OPTMV) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -DSKETCH_FLOAT_TYPE="float" -DNDEBUG -std=c11
%.ld64o: %.cpp $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -DSKETCH_FLOAT_TYPE="long double" -DNDEBUG -flto  -DLSHIDTYPE="uint64_t"
%.f64o: %.cpp $(wildcard src/*.h)
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -c -o $@ $(LIB) $(EXTRA) -DSKETCH_FLOAT_TYPE="float" -DNDEBUG  -flto -DLSHIDTYPE="uint64_t"

mmtest: test/mmtest.cpp src/mmvec.h
	$(CXX) $(INC) $(OPT) $(WARNING) $(MACH) $< -o $@ $(LIB) $(EXTRA)


BWF=libBigWig/bwRead.o libBigWig/bwStats.o libBigWig/bwValues.o libBigWig/bwWrite.o libBigWig/io.o
bwf:
	echo $(BWF)

libgomp.a:
	ln -sf $(shell $(CXX) --print-file-name=libgomp.a)
dashing2_s128: $(D2SRC) $(wildcard src/*.h) libgomp.a $(BWF)
	$(CXX) $(CXXFLAGS) $(OPT) $(WARNING) $(MACH) $(INC) $(LIB) -mno-avx512dq -mno-avx512vl -mno-avx512f -mno-avx512bw -mno-avx -mno-avx2 -msse2 -msse4.1 -static-libstdc++ -static-libgcc -flto \
    libgomp.a $(BWF) \
		-DNDEBUG $(D2SRC) -o $@ $(EXTRA) $(LIB) -ldl -lz -O0

dashing2_savx: $(D2SRC) $(wildcard src/*.h) libgomp.a $(BWF)
	$(CXX) $(CXXFLAGS) $(OPT) $(WARNING) $(MACH) $(INC) $(LIB) -mno-avx512dq -mno-avx512vl -mno-avx512f -mno-avx512bw -mavx -mno-avx2 -msse2 -msse4.1 -static-libstdc++ -static-libgcc -flto \
    libgomp.a $(BWF) \
		-DNDEBUG $(D2SRC) -o $@ $(EXTRA) $(LIB) -ldl -lz

dashing2_savx2: $(D2SRC) $(wildcard src/*.h) libgomp.a $(BWF)
	$(CXX) $(CXXFLAGS) $(OPT) $(WARNING) $(MACH) $(INC) $(LIB) -mno-avx512dq -mno-avx512vl -mno-avx512f -mno-avx512bw -mavx -mavx2 -msse2 -msse4.1 -static-libstdc++ -static-libgcc -flto \
    libgomp.a $(BWF) \
		-DNDEBUG $(D2SRC) -o $@ $(EXTRA) $(LIB) -ldl -lz

dashing2_s512: $(D2SRC) $(wildcard src/*.h) libgomp.a $(BWF)
	$(CXX) $(CXXFLAGS) $(OPT) $(WARNING) $(MACH) $(INC) $(LIB) -mno-avx512dq -mno-avx512vl -mno-avx512bw -mavx512f -mavx -mavx2 -msse2 -msse4.1 -static-libstdc++ -static-libgcc -flto \
    libgomp.a $(BWF) \
		-DNDEBUG $(D2SRC) -o $@ $(EXTRA) $(LIB) -ldl -lz

dashing2_s512bw: $(D2SRC) $(wildcard src/*.h) libgomp.a $(BWF)
	$(CXX) $(CXXFLAGS) $(OPT) $(WARNING) $(MACH) $(INC) $(LIB) -mavx512dq -mavx512vl -mavx512bw -mavx512f -mavx -mavx2 -msse2 -msse4.1 -static-libstdc++ -static-libgcc -flto \
    libgomp.a $(BWF) -DNDEBUG $(D2SRC) -o $@ $(EXTRA) $(LIB) -ldl -lz -O0

dashing2_static: dashing2_s128 dashing2_savx dashing2_savx2 dashing2_s512 dashing2_s512bw

libBigWig/%.o: libBigWig/%.c libBigWig.a
	cd libBigWig && make $(shell basename $@)



libBigWig.a: $(wildcard libBigWig/*.c) $(wildcard libBigWig/*.h)
	cd libBigWig && sed -i $(SEDSTR) 's/HAVE_CURL:/#/' Makefile && $(MAKE) && cp libBigWig.a ..

test: readfx readbw

clean:
	rm -f dashing2 dashing2-ld dashing2-f libBigWig.a $(OBJ) $(OBJLD) $(OBJF) readfx readfx-f readfx-ld readbw readbw readbw-f readbw-ld src/*.0 src/*.do src/*.fo src/*.gobj src/*.ldo src/*.0\
		src/*.vo src/*.sano src/*.ld64o src/*.f64o src/*.64o
