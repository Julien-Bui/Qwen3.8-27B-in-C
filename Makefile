CC      = gcc
CFLAGS  = -Wall -Wextra -O2 -std=c11 -Iinclude -march=native
LDLIBS  = -lm

all: build/gguf_test build/test_quant build/test_model build/test_kernels build/test_ops

build:
	mkdir -p build

build/gguf_test: src/gguf.c src/main.c include/gguf.h | build
	$(CC) $(CFLAGS) src/gguf.c src/main.c -o $@ $(LDLIBS)

build/test_quant: src/gguf.c src/quant.c src/test_quant.c include/gguf.h include/quant.h | build
	$(CC) $(CFLAGS) src/gguf.c src/quant.c src/test_quant.c -o $@ $(LDLIBS)

build/test_model: src/gguf.c src/quant.c src/model.c src/test_model.c include/gguf.h include/quant.h include/model.h | build
	$(CC) $(CFLAGS) src/gguf.c src/quant.c src/model.c src/test_model.c -o $@ $(LDLIBS)

build/test_kernels: src/gguf.c src/quant.c src/model.c src/kernels.c src/threadpool.c src/test_kernels.c include/gguf.h include/quant.h include/model.h include/kernels.h include/threadpool.h | build
	$(CC) $(CFLAGS) src/gguf.c src/quant.c src/model.c src/kernels.c src/threadpool.c src/test_kernels.c -o $@ $(LDLIBS)

build/test_ops: src/kernels.c src/quant.c src/threadpool.c src/test_ops.c include/kernels.h include/quant.h include/threadpool.h | build
	$(CC) $(CFLAGS) src/kernels.c src/quant.c src/threadpool.c src/test_ops.c -o $@ $(LDLIBS)

clean:
	rm -rf build

.PHONY: all clean
