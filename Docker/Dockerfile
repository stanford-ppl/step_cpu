FROM ubuntu:24.04
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    build-essential ca-certificates cmake ninja-build git wget \
    python3-dev python3-venv pkg-config \
    libopenblas-dev libstdc++-14-dev libzip-dev \
    sudo swig time

ARG CLANG_VER=19
RUN apt-get install -y clang-${CLANG_VER} libomp-${CLANG_VER}-dev && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-${CLANG_VER} 100 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-${CLANG_VER} 100 && \
    update-alternatives --install /usr/bin/llvm-ar llvm-ar  /usr/bin/llvm-ar-${CLANG_VER} 100 && \
    update-alternatives --install /usr/bin/llvm-profdata llvm-profdata  /usr/bin/llvm-profdata-${CLANG_VER} 100

RUN useradd -l -ms /bin/bash dockeruser && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    adduser dockeruser sudo && \
    echo 'dockeruser:dockeruser' | chpasswd
USER dockeruser
WORKDIR /home/dockeruser

ARG PT_VER=2.8.0
ARG PTHASH=ba56102387ef21a3b04b357e5b183d48f0afefc7

RUN python3 -m venv mochaenv && \
    source mochaenv/bin/activate && \
    pip install --upgrade pip && \
    git clone https://github.com/pytorch/pytorch && \
    cd pytorch && \
    git checkout ${PTHASH} -b ${PT_VER} && \
    git submodule sync && git submodule update --init --recursive

ENV CC=/usr/bin/clang CXX=/usr/bin/clang++ \
    GXX_INC=/usr/include/c++/14 \
    LLVM_INC=/usr/lib/llvm-${CLANG_VER}/lib/clang/${CLANG_VER}/include \
    GXX_LIB=/usr/lib/gcc/x86_64-linux-gnu/14 \
    LLVM_LIB=/usr/lib/llvm-${CLANG_VER}/lib

ENV CXXFLAGS="-fclang-abi-compat=17 -Wno-unknown-warning-option -stdlib=libstdc\\+\\+ -I${GXX_INC} -I${LLVM_INC}" \
    LDFLAGS="-stdlib=libstdc\\+\\+ -L${GXX_LIB} -Wl,-rpath,${GXX_LIB} -L${LLVM_LIB} -Wl,-rpath,${LLVM_LIB}"

ENV BUILD_TEST=0 USE_CUDA=0 USE_CUDNN=0 USE_DISTRIBUTED=0 USE_NCCL=0 USE_TENSORPIPE=0
ENV PIP_NO_BINARY="numpy,opencv-python,torch"
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

# transformers only
ENV PIP_NO_BINARY="${PIP_NO_BINARY},transformers"
ARG HF_VER=v4.57.1
RUN source mochaenv/bin/activate && \
    git clone https://github.com/huggingface/transformers.git && \
    cd transformers && \
    git checkout tags/${HF_VER} -b ${HF_VER} && \
    python3 -m pip install -e .[torch]

ADD --chown=dockeruser:dockeruser ./scripts ./scripts
COPY --chown=dockeruser:dockeruser ./scripts/bash_aliases /home/dockeruser/.bash_aliases
COPY --chown=dockeruser:dockeruser ./scripts/run_all_tests.sh /home/dockeruser/
USER root
ENTRYPOINT ["./scripts/entrypoint.sh"]