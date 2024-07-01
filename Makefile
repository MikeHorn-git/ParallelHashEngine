CC = nvcc
DEBUG_FLAGS = -g -G

all: md5

md5:
	$(CC) md5.cu -o md5

debug_md5:
	$(CC) $(DEBUG_FLAGS) md5.cu -o debug_md5

clean:
	rm -rf md5 debug_md5

.PHONY: all md5 debug_md5 clean
