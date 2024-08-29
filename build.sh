export NVCC_PREPEND_FLAGS="--forward-unknown-opts"

export TORCH_CUDA_ARCH_LIST="7.0 7.2 7.5 8.0 8.6 8.7"

SAM2_BUILD_CUDA=1 SAM2_BUILD_ALLOW_ERRORS=1 MAX_JOBS=8 python setup.py build_ext




CUDA_VISIBLE_DEVICES=1 python INFERENCE.py \
/lscratch/34819899/pannuke/fold1/images/1_1790.png \
/lscratch/34819899/pannuke/fold1/labels/1_1790.npy


CUDA_VISIBLE_DEVICES=1 python TEST_Net.py
