CC=icx
CFLAGS=-O2 -fiopenmp -fsycl
LDFLAGS=-lpng -lm

all: stegano


stegano: main.o stegano_routines.o io_routines.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f stegano *.o *.opt.yaml


.PHONY: clean all
