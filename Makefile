COMPILER=nvcc
COMPILER_FLAGS=-O3

build: src/main.cu
	$(COMPILER) $(COMPILER_FLAGS) src/main.cu -o image_processor

clean:
	rm -f image_processor
