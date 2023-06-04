# Binôme : Maxime Moshfeghi & Samy Afker

# Définition des Variables (à compléter)
CC = c99
CFLAGS = -Wextra -pedantic -c
LDFLAGS = 
EXE = 

# Règles (à compléter avec des variables automatiques)
all: test_file exemple_file

test_file: test_file.o file.o
	$(CC) $(LDFLAGS) $^ -o $@

exemple_file: exemple_file.o file.o
	$(CC) $(LDFLAGS) $^ -o $@

file.o: file.c file.h
	$(CC) $(CFLAGS) $<

test_file.o: test_file.c
	$(CC) $(CFLAGS) $<

exemple_file.o: exemple_file.c
	$(CC) $(CFLAGS) $<

# Il y avait makedepend 
depend:
	makedepend *.c -Y

clean:
	rm *.o test_file exemple_file

.PHONY: clean all makedepend

