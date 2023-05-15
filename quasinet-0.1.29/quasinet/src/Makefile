CC = gcc
#CC = x86_64-w64-mingw32-gcc
#CFLAGS = -O3 -march=native -ffast-math -fPIC
CFLAGS = -O3 -ffast-math -fPIC

LDFLAGS = -shared
DCOR_LIB_NAME = dcor.so
CFUNC_LIB_NAME = Cfunc.so
OMP_FLAGS = -fopenmp

all: $(DCOR_LIB_NAME) $(CFUNC_LIB_NAME)

$(DCOR_LIB_NAME): dcor.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $(DCOR_LIB_NAME) dcor.c

$(CFUNC_LIB_NAME): Cfunc.c
	$(CC) $(CFLAGS) $(LDFLAGS) $(OMP_FLAGS) -o $(CFUNC_LIB_NAME) Cfunc.c

clean:
	rm -f $(DCOR_LIB_NAME) $(CFUNC_LIB_NAME)
