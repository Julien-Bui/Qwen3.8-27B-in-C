CC      = gcc
CFLAGS  = -Wall -Wextra -O3 -std=c11 -Iinclude -march=native -mavx2 -mfma
LDLIBS  = -lm -lpthread

SRCS_COMMON = src/gguf.c src/quant.c src/kernels.c src/threadpool.c src/model.c src/tokenizer.c src/sampler.c

all: build/qwen build/test_tokenizer build/test_forward build/test_ops build/test_kernels build/test_quant build/test_model

build:
	mkdir -p build

build/qwen: src/main.c $(SRCS_COMMON) | build
	$(CC) $(CFLAGS) src/main.c $(SRCS_COMMON) -o $@ $(LDLIBS)

build/test_tokenizer: src/test_tokenizer.c src/tokenizer.c src/gguf.c | build
	$(CC) $(CFLAGS) src/test_tokenizer.c src/tokenizer.c src/gguf.c -o $@ $(LDLIBS)

build/test_forward: src/test_forward.c $(SRCS_COMMON) | build
	$(CC) $(CFLAGS) src/test_forward.c $(SRCS_COMMON) -o $@ $(LDLIBS)

build/test_model: src/test_model.c src/gguf.c src/quant.c src/model.c src/kernels.c src/threadpool.c | build
	$(CC) $(CFLAGS) src/test_model.c src/gguf.c src/quant.c src/model.c src/kernels.c src/threadpool.c -o $@ $(LDLIBS)

build/test_kernels: src/test_kernels.c src/gguf.c src/quant.c src/model.c src/kernels.c src/threadpool.c | build
	$(CC) $(CFLAGS) src/test_kernels.c src/gguf.c src/quant.c src/model.c src/kernels.c src/threadpool.c -o $@ $(LDLIBS)

build/test_ops: src/test_ops.c src/kernels.c src/quant.c src/threadpool.c | build
	$(CC) $(CFLAGS) src/test_ops.c src/kernels.c src/quant.c src/threadpool.c -o $@ $(LDLIBS)

build/test_quant: src/test_quant.c src/gguf.c src/quant.c | build
	$(CC) $(CFLAGS) src/test_quant.c src/gguf.c src/quant.c -o $@ $(LDLIBS)

clean:
	rm -rf build

serve: build/qwen
	python3 server.py --port 8080

.PHONY: all clean serve
