SHELL = bash

CXX = clang++ -std=c++23
CFLAGS = -O3 -ffast-math -march=native -fvisibility=hidden -DPROGRESS

CFLAGS += -D SIMULATION_MODE_FM
#CFLAGS += -D SIMULATION_MODE_AFM
#CFLAGS += -D SIMULATION_MODE_SG

MULTI := 1
L := 10
burn_in := 1000
calc_steps := 1000

oneapi_device_selector = opencl:cpu
targets = spir64_x86_64
opts =
ifeq ($(shell grep -q avx512 /proc/cpuinfo && echo true || echo false),true)
	opts += -Xs "-march=avx512"
else ifeq ($(shell grep -q avx2 /proc/cpuinfo && echo true || echo false),true)
	opts += -Xs "-march=avx2"
endif

has_iris := $(shell sycl-ls --ignore-device-selectors | grep -q level_zero && echo true || echo false)
ifeq ($(has_iris),true)
	oneapi_device_selector = level_zero:gpu
	targets = spir64_gen
	target = $(shell sycl-ls --ignore-device-selectors | grep -Po '^\[level_zero:gpu\].+ \K[0-9\.]+(?= \[.+\]$$)')
	opts = -Xs "-device $(target)"
endif

has_gpu := $(shell nvidia-smi >&/dev/null && echo true || echo false)
ifeq ($(has_gpu),true)
	oneapi_device_selector = cuda:*
	targets := $(targets),nvptx64-nvidia-cuda
	cuda_home = $(shell printenv CUDA_HOME)
	gpu_arch = $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sed 's/\.//g' | head -n1)
	opts += --cuda-path=$(cuda_home) --cuda-include-ptx=sm_$(gpu_arch)
endif

CFLAGS += -D ONEAPI_DEVICE_SELECTOR=\"$(oneapi_device_selector)\"

main_sycl: main_sycl.cpp Makefile
	$(CXX) $(CFLAGS) -fsycl -fsycl-targets=$(targets) $(opts) -fuse-ld=lld -o $@ $<

run: main_sycl
	$(eval T := $(shell echo "5 4 3 2"; yes 1 | head -n $$((32 * $(MULTI) - 4)) | tr '\n' ' '))
	printf '%s\n' 'T: $(T)' 'calc_steps: $(calc_steps)' 'burn_in: $(burn_in)' 'run_simulation' | MULTI=$(MULTI) ./main_sycl $(L) | grep -Po 'e1":\[\K[^\]]+'

clean:
	rm -f main_sycl
