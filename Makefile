all: cugemm.bin cugemm-debug.bin cugemm-profile.bin

# SOURCE_FILE=cugemm.cu
SOURCE_FILE=cugemm-hw4.cu

# optimized binary
cugemm.bin: $(SOURCE_FILE)
	nvcc -std=c++17 --generate-code=arch=compute_61,code=[compute_61,sm_61] $^ -lcublas -o $@

# debug binary without optimizations
cugemm-debug.bin: $(SOURCE_FILE)
	nvcc -g -G -src-in-ptx -std=c++17 --generate-code=arch=compute_61,code=[compute_61,sm_61] $^ -lcublas -o $@

# optimized binary with line number information for profiling
cugemm-profile.bin: $(SOURCE_FILE)
	nvcc -g --generate-line-info -src-in-ptx -std=c++17 --generate-code=arch=compute_61,code=[compute_61,sm_61] $^ -lcublas -o $@

# NB: make sure you change the --algo flag here to profile the one you care about. 
# You can change the --export flag to set the filename of the profiling report that is produced.
profile-ncu: cugemm-profile.bin
	sudo ncu --export my-profile --set full ./cugemm-profile.bin --size=1024 --reps=1 --algo=1 --validate=false

# profile with Nsight Systems to see a nice timeline of memory copies and kernel execution
profile-nsys: cugemm.bin
	nsys profile -t cuda,osrt,nvtx -o myprofile ./cugemm.bin --validate=false --algo 6 --size 8192 --streams 4

clean:
	rm -f cugemm*.bin
