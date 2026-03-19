# Changed base image as we only use CPU
FROM ubuntu:24.04

SHELL ["/bin/bash", "-c"]

# run apt update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    build-essential \
    git \
    libopenblas-dev \
    libstdc++-14-dev \
    libzip-dev \
    pkg-config \
    python3-dev \
    python3-venv \
    sudo \
    swig \
    time \
    wget \
    ca-certificates \
    cmake \
    ninja-build

#ARG CLANG_VER=16 # works
#ARG CLANG_VER=17 # works, -Wno-unknown-warning-option
#ARG CLANG_VER=18 # -fclang-abi-compat=17 added to CXXFLAGS & TORCH_NVCC_FLAGS
ARG CLANG_VER=19
#ARG CLANG_VER=20 # "clang version must be less than 20 and greater than 3.2"
RUN apt-get install -y \
    clang-${CLANG_VER} libomp-${CLANG_VER}-dev && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-${CLANG_VER} 100 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${CLANG_VER} 100 && \
    update-alternatives --install /usr/bin/llvm-ar llvm-ar  /usr/bin/llvm-ar-${CLANG_VER} 100 && \
    update-alternatives --install /usr/bin/llvm-profdata llvm-profdata  /usr/bin/llvm-profdata-${CLANG_VER} 100

# set up a user
RUN useradd -l -ms /bin/bash dockeruser && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    adduser dockeruser sudo
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN echo 'dockeruser:dockeruser' | chpasswd
USER dockeruser
ARG DOCKERUSERHOME=/home/dockeruser
WORKDIR ${DOCKERUSERHOME}

# get pytorch
#ARG PT_VER=2.9.1
#ARG PTHASH=d38164a545b4a4e4e0cf73ce67173f70574890b6
ARG PT_VER=2.8.0
ARG PTHASH=ba56102387ef21a3b04b357e5b183d48f0afefc7
RUN python3 -m venv mochaenv && \
    source mochaenv/bin/activate && \
    pip install --upgrade pip && \
    git clone https://github.com/pytorch/pytorch && \
    cd pytorch && \
    git checkout ${PTHASH} -b ${PT_VER} && \
    git submodule sync && \
    git submodule update --init --recursive

# set compiler and flags to link libstdc++ here
# need a new ENV to use definitions
ENV CC=/usr/bin/clang CXX=/usr/bin/clang++ \
    GXX_INC=/usr/include/c++/14 \
    LLVM_INC=/usr/lib/llvm-${CLANG_VER}/lib/clang/${CLANG_VER}/include \
    GXX_LIB=/usr/lib/gcc/x86_64-linux-gnu/14 \
    LLVM_LIB=/usr/lib/llvm-${CLANG_VER}/lib
ENV CXXFLAGS="-fclang-abi-compat=17 -Wno-unknown-warning-option -stdlib=libstdc\\+\\+ -I${GXX_INC} -I${LLVM_INC}" \
    LDFLAGS="-stdlib=libstdc\\+\\+ -L${GXX_LIB} -Wl,-rpath,${GXX_LIB} -L${LLVM_LIB} -Wl,-rpath,${LLVM_LIB}"

# cuda variables => deleted

ENV BUILD_TEST=0 \
    USE_CUDA=0 \
    USE_CUDNN=0 \
    USE_DISTRIBUTED=0 \
    USE_NCCL=0 \
    USE_TENSORPIPE=0

# build pytorch
# list pip dependencies that should be built from source
ENV PIP_NO_BINARY="numpy, \
    opencv-python, \
    torch"
# -j$(nproc) runs out of memory. Setting NJOBS=12 peaks at about 40GB.
ARG NJOBS=12
RUN source mochaenv/bin/activate && \
    cd pytorch && \
    python3 -m pip install -r requirements.txt && \
    python3 setup.py build --cmake-only && \
    cd build && \
    ninja torch_cpu c10 -j${NJOBS} && \
    ninja -j${NJOBS} && \
    cd .. && \
    python3 setup.py bdist_wheel && \
    python3 -m pip install dist/torch-${PT_VER}*.whl

# use this to test in container
# python3 -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda); print(torch.cuda.device_count()); print(torch.cuda.is_available()); print(torch.backends.cudnn.is_available()); print(torch.backends.cudnn.enabled); print(torch.backends.cudnn.version())"

# some additional pip packages for benchmarking
RUN source mochaenv/bin/activate && \
    python3 -m pip install \
    datasets \
    nvidia-ml-py \
    psutil \
    pytest \
    tabulate

# build torchvision => deleted

# build ultralytics => deleted
# versions of tensorrt depend on compute capability. Enabling sm_70 below.
# pip index versions tensorrt-cu12

# build huggingface transformers
ENV PIP_NO_BINARY="${PIP_NO_BINARY},transformers"
ARG HF_VER=v4.57.1
RUN source mochaenv/bin/activate && \
    git clone https://github.com/huggingface/transformers.git && \
    cd transformers && \
    git checkout tags/${HF_VER} -b ${HF_VER} && \
    python3 -m pip install -e .[torch]

# build faiss and hnswlib => deleted


# build darknet (yolov4) => deleted
# the feature/clang branch on this fork changes gcc to clang
# it is a few commits beyond 59596d7880f6504768df41d6daa586f5cb2b932f


# build or-tools => deleted
# hash of stable branch in Jan. 26

# install pytorch_geometric and build external libraries => deleted

# add files. .bash_aliases gets sourced from the su command in entrypoint.sh
ADD --chown=dockeruser:dockeruser ./scripts ./scripts
COPY --chown=dockeruser:dockeruser ./scripts/bash_aliases /home/dockeruser/.bash_aliases
COPY --chown=dockeruser:dockeruser ./scripts/run_all_tests.sh /home/dockeruser/

# go in as root and change user in entrypoint.sh
USER root
# set entrypoint
ENTRYPOINT ["./scripts/entrypoint.sh"]