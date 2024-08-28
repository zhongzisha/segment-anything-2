export NVCC_PREPEND_FLAGS="--forward-unknown-opts"

export TORCH_CUDA_ARCH_LIST="7.0 7.2 7.5 8.0 8.6 8.7"

SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=1 MAX_JOBS=8 python setup.py build_ext








