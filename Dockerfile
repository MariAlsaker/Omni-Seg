FROM ubuntu:16.04
RUN set -xe 		\
    && echo '#!/bin/sh' > /usr/sbin/policy-rc.d 	\
    && echo 'exit 101' >> /usr/sbin/policy-rc.d 	\
    && chmod +x /usr/sbin/policy-rc.d 		\
    && dpkg-divert --local --rename --add /sbin/initctl 	\
    && cp -a /usr/sbin/policy-rc.d /sbin/initctl 	\
    && sed -i 's/^exit.*/exit 0/' /sbin/initctl 		\
    && echo 'force-unsafe-io' > /etc/dpkg/dpkg.cfg.d/docker-apt-speedup 		\
    && echo 'DPkg::Post-Invoke { "rm -f /var/cache/apt/archives/*.deb /var/cache/apt/archives/partial/*.deb /var/cache/apt/*.bin || true"; };' > /etc/apt/apt.conf.d/docker-clean 	\
    && echo 'APT::Update::Post-Invoke { "rm -f /var/cache/apt/archives/*.deb /var/cache/apt/archives/partial/*.deb /var/cache/apt/*.bin || true"; };' >> /etc/apt/apt.conf.d/docker-clean 	\
    && echo 'Dir::Cache::pkgcache ""; Dir::Cache::srcpkgcache "";' >> /etc/apt/apt.conf.d/docker-clean 		\
    && echo 'Acquire::Languages "none";' > /etc/apt/apt.conf.d/docker-no-languages 		\
    && echo 'Acquire::GzipIndexes "true"; Acquire::CompressionTypes::Order:: "gz";' > /etc/apt/apt.conf.d/docker-gzip-indexes 		\
    && echo 'Apt::AutoRemove::SuggestsImportant "false";' > /etc/apt/apt.conf.d/docker-autoremove-suggests
RUN rm -rf /var/lib/apt/lists/*
RUN mkdir -p /run/systemd \
    && echo 'docker' > /run/systemd/container
CMD ["/bin/bash"]
ENV NVARCH=x86_64
ENV NVIDIA_REQUIRE_CUDA=cuda>=9.0
ENV NV_CUDA_CUDART_VERSION=9.0.176-1
ENV NV_ML_REPO_ENABLED=1
ENV NV_ML_REPO_URL=https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64
ARG TARGETARCH TARGETARCH
RUN apt-get update \
    && apt-get install -y --no-install-recommends     ca-certificates apt-transport-https gnupg-curl \
    &&     NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 \
    &&     NVIDIA_GPGKEY_FPR=ae09fe4bbd223a84b2ccfce3f60f4b3d7fa2af80 \
    &&     apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/${NVARCH}/7fa2af80.pub \
    &&     apt-key adv --export --no-emit-version -a $NVIDIA_GPGKEY_FPR | tail -n +5 > cudasign.pub \
    &&     echo "$NVIDIA_GPGKEY_SUM  cudasign.pub" | sha256sum -c --strict - \
    && rm cudasign.pub \
    &&     echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list \
    &&     if [ ! -z ${NV_ML_REPO_ENABLED} ]; then echo "deb ${NV_ML_REPO_URL} /" > /etc/apt/sources.list.d/nvidia-ml.list; fi \
    &&     apt-get purge --auto-remove -y gnupg-curl     \
    && rm -rf /var/lib/apt/lists/*
ENV CUDA_VERSION=9.0.176
ARG TARGETARCH
RUN apt-get update \
    && apt-get install -y --allow-unauthenticated --no-install-recommends     cuda-cudart-9-0=${NV_CUDA_CUDART_VERSION}     \
    && ln -s cuda-9.0 /usr/local/cuda \
    &&     rm -rf /var/lib/apt/lists/*
LABEL com.nvidia.volumes.needed=nvidia_driver
LABEL com.nvidia.cuda.version=9.0.176
ARG TARGETARCH
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
    &&     echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
COPY NGC-DL-CONTAINER-LICENSE /
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV NV_CUDA_LIB_VERSION=9.0.176-1
ENV NV_NVTX_VERSION=9.0.176-1
ENV NV_LIBNPP_VERSION=9.0.176-1
ENV NV_LIBCUSPARSE_VERSION=9.0.176-1
ENV NV_LIBCUBLAS_PACKAGE_NAME=cuda-cublas-9-0
ENV NV_LIBCUBLAS_VERSION=9.0.176-1
ENV NV_LIBCUBLAS_PACKAGE=cuda-cublas-9-0=9.0.176-1
ARG TARGETARCH
RUN apt-get update \
    && apt-get install -y --allow-unauthenticated --no-install-recommends     cuda-libraries-9-0=${NV_CUDA_LIB_VERSION}     cuda-npp-9-0=${NV_LIBNPP_VERSION}     cuda-cusparse-9-0=${NV_LIBCUSPARSE_VERSION}     ${NV_LIBCUBLAS_PACKAGE}     \
    && rm -rf /var/lib/apt/lists/*
ARG TARGETARCH
RUN apt-mark hold ${NV_LIBNCCL_PACKAGE_NAME} ${NV_LIBCUBLAS_PACKAGE_NAME}
ENV NV_CUDA_LIB_VERSION=9.0.176-1
ENV NV_CUDA_CUDART_DEV_VERSION=9.0.176-1
ENV NV_NVML_DEV_VERSION=9.0.176-1
ENV NV_LIBCUSPARSE_DEV_VERSION=9.0.176-1
ENV NV_LIBNPP_DEV_VERSION=9.0.176-1
ENV NV_LIBCUBLAS_DEV_PACKAGE_NAME=cuda-cublas-dev-9-0
ENV NV_LIBCUBLAS_DEV_VERSION=9.0.176-1
ENV NV_LIBCUBLAS_DEV_PACKAGE=cuda-cublas-dev-9-0=9.0.176-1
ARG TARGETARCH
RUN apt-get update \
    && apt-get install -y --allow-unauthenticated --no-install-recommends     cuda-nvml-dev-9-0=${NV_NVML_DEV_VERSION}     cuda-command-line-tools-9-0=${NV_CUDA_LIB_VERSION}     cuda-npp-dev-9-0=${NV_LIBNPP_DEV_VERSION}     cuda-libraries-dev-9-0=${NV_CUDA_LIB_VERSION}     cuda-minimal-build-9-0=${NV_CUDA_LIB_VERSION}     ${NV_LIBCUBLAS_DEV_PACKAGE}     ${NV_LIBNCCL_DEV_PACKAGE}     \
    && rm -rf /var/lib/apt/lists/*
ARG TARGETARCH
RUN apt-mark hold ${NV_LIBCUBLAS_DEV_PACKAGE_NAME} ${NV_LIBNCCL_DEV_PACKAGE_NAME}
ENV LIBRARY_PATH=/usr/local/cuda/lib64/stubs
ENV NV_CUDNN_PACKAGE_VERSION=7.6.4.38-1
ENV NV_CUDNN_VERSION=7.6.4.38
ENV NV_CUDNN_PACKAGE_NAME=libcudnn7
ENV NV_CUDNN_PACKAGE=libcudnn7=7.6.4.38-1+cuda9.0
ENV NV_CUDNN_PACKAGE_DEV=libcudnn7-dev=7.6.4.38-1+cuda9.0
ARG TARGETARCH
ENV CUDNN_VERSION=7.6.4.38
LABEL com.nvidia.cudnn.version=7.6.4.38
ARG TARGETARCH
RUN apt-get update \
    && apt-get install -y --allow-unauthenticated --no-install-recommends     ${NV_CUDNN_PACKAGE}     ${NV_CUDNN_PACKAGE_DEV}     \
    && apt-mark hold ${NV_CUDNN_PACKAGE_NAME} \
    &&     rm -rf /var/lib/apt/lists/*
RUN apt-get -y update \
    &&     apt-get -y upgrade \
    &&     apt-get -y install wget bc \
    &&     apt-get -y install --no-install-recommends apt-utils \
    &&     apt-get -y install libsm6 libxrender1 libfontconfig1 \
    &&     apt-get -y install openslide-tools \
    &&     apt-get -y install python3-openslide
RUN mkdir /INPUTS \
    &&     mkdir /OUTPUTS
RUN mkdir /tmp/miniconda \
    &&    cd /tmp/miniconda \
    &&    apt-get install bzip2 \
    &&    wget -nv https://repo.continuum.io/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh  --no-check-certificate \
    &&    chmod +x Miniconda3-py37_4.12.0-Linux-x86_64.sh \
    &&    bash Miniconda3-py37_4.12.0-Linux-x86_64.sh -b -p /pythondir/miniconda \
    &&    rm -r /tmp/miniconda
RUN /pythondir/miniconda/bin/pip3 install numpy==1.19.5 \
    &&    /pythondir/miniconda/bin/pip3 install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html \
    &&    /pythondir/miniconda/bin/pip3 install torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html \
    &&    /pythondir/miniconda/bin/pip3 install imgaug==0.4.0 \
    &&    /pythondir/miniconda/bin/pip3 install tensorboardX==2.4 \
    &&    /pythondir/miniconda/bin/pip3 install SimpleITK==2.1.1 \
    &&    /pythondir/miniconda/bin/pip3 install xmltodict==0.12.0 \
    &&    /pythondir/miniconda/bin/pip3 install openslide-python==1.1.2 \
    &&    /pythondir/miniconda/bin/pip3 install scipy==1.5.4 \
    &&    /pythondir/miniconda/bin/pip3 install opencv-python==3.4.2.17 \
    &&    /pythondir/miniconda/bin/pip3 install matplotlib==3.3.4 \
    &&    /pythondir/miniconda/bin/pip3 install scikit-image==0.17.2 \
    &&    /pythondir/miniconda/bin/pip3 install Pillow==8.3.2 \
    &&    /pythondir/miniconda/bin/pip3 install argparse==1.4.0 \
    &&    /pythondir/miniconda/bin/pip3 install pandas==1.1.5 \
    &&    /pythondir/miniconda/bin/pip3 install nibabel==3.2.1 \
    &&    /pythondir/miniconda/bin/pip3 install scikit-learn==0.24.2
ENV PATH=/pythondir/miniconda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV CONDA_DEFAULT_ENV=python37
ENV CONDA_PREFIX=/pythondir/miniconda/envs/python37
COPY Omni_seg_pipeline_gpu /Omni-Seg/Omni_seg_pipeline_gpu/
COPY running_all.sh /Omni-Seg/Omni_seg_pipeline_gpu/
RUN chmod a+x /Omni-Seg/Omni_seg_pipeline_gpu/running_all.sh 
RUN python3 /Omni-Seg/Omni_seg_pipeline_gpu/apex/setup.py install
CMD ["./Omni-Seg/Omni_seg_pipeline_gpu/running_all.sh"]
