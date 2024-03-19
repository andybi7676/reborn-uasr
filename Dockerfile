FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-devel
# establish the environment variables and intall the required packages
ENV APT_INSTALL="apt-get install -y --no-install-recommends"
ENV DEBIAN_FRONTEND=noninteractive
RUN rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        wget \
        git \
        g++ \
        cmake \
        # for MKL
        # apt-transport-https gpg-agent gnupg2 \
        # for kenlm
        libboost-thread-dev libboost-test-dev libboost-system-dev libboost-program-options-dev zlib1g-dev libbz2-dev liblzma-dev \
        # for arrayfire
        # libboost-stacktrace-dev \
        # FFTW
        libfftw3-dev \
        # ssh for OpenMPI
        openssh-server openssh-client \
        # for OpenMPI
        # libopenmpi-dev openmpi-bin \
        # for kaldi
        automake autoconf unzip sox gfortran libtool subversion python2.7 intel-mkl && \
# ==================================================================
# clean up everything
# ------------------------------------------------------------------
        apt-get clean && \
        apt-get -y autoremove && \
        rm -rf /var/lib/apt/lists/* && \
        apt clean 
# Set the working directory in the container
WORKDIR /workspace
# add current directory to the container
ADD . reborn-uasr
# install required python packages
RUN pip install -r reborn-uasr/requirements.txt
# install fairseq based on current version
RUN pip install -e reborn-uasr/fairseq
ENV FAIRSEQ_ROOT=/workspace/reborn-uasr/fairseq
# install kenlm
RUN git clone https://github.com/kpu/kenlm.git && \
    cd kenlm && mkdir -p build && \
    cd build && cmake .. && \
    make -j 4 && \
    pip install /workspace/kenlm
ENV KENLM_ROOT=/workspace/kenlm
# install flashlight python bindings (text and sequence lib are currently separated from the main repo. \
# However, the sequence lib's python binding is currently not available. Switch to v0.3.2 as a walkaround)
RUN git clone https://github.com/flashlight/flashlight.git && cd flashlight && git checkout v0.3.2 && \
    cd bindings/python && conda install mkl-include -y && \
    pip install . 
# install kaldi and pykaldi based on .whl packages
# install pykaldi, the .whl.gz is from https://github.com/pykaldi/pykaldi/releases
RUN git clone https://github.com/pykaldi/pykaldi.git && cd pykaldi && mv /workspace/reborn-uasr/pykaldi-0.2.2-cp310-cp310-linux_x86_64.whl.gz . && \
    gzip -d pykaldi-0.2.2-cp310-cp310-linux_x86_64.whl.gz && pip install pykaldi-0.2.2-cp310-cp310-linux_x86_64.whl
# install kaldi
ENV KALDI_ROOT=/workspace/kaldi
RUN cp /workspace/pykaldi/tools/install_kaldi.sh /workspace && \
    ./install_kaldi.sh && tail -n +2 /workspace/pykaldi/tools/path.sh >> ~/.bashrc && \
    rm /workspace/install_kaldi.sh 
# preparation for kaldi-hmm-self-training
RUN pip install kaldi-io && \
    # the kaldi-io is required if you want to do kaldi-hmm-self-training (see s2p/kaldi_self_train/st/local/prepare_data_from_w2v.py)
    cp -r /workspace/reborn-uasr/s2p/kaldi_self_train /workspace/kaldi/egs/ && cd /workspace/kaldi/egs && \
    ln -s ../../wsj/s5/steps steps && ln -s ../../wsj/s5/utils utils
# CMD is bash
CMD ["/bin/bash"]

