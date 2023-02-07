CC=icx
CFLAGS=-O2 -std=c99 -fiopenmp
LDFLAGS=-lpng -lm

all: steano


steano: main.o steano_routines.o io_routines.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f steano *.o *.opt.yaml


.PHONY: clean all
