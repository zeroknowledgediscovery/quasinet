CC = gcc
CFLAGS = -O3 -march=native -fPIC
OMPFLAGS = -fopenmp
LIBFLAGS = -shared
LDFLAGS = -L. -ldcor

all: Cfunc.so dcor.so test_dcor

Cfunc.so: Cfunc.c
	$(CC) $(CFLAGS) $(OMPFLAGS) $(LIBFLAGS) -o $@ $<

dcor.so: dcor.c
	$(CC) $(CFLAGS) -ffast-math $(LIBFLAGS) -o $@ $<

test_dcor: test_dcor.c dcor.so
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f Cfunc.so dcor.so test_dcor
