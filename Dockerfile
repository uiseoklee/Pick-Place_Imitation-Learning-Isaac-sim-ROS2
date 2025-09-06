FROM pytorch/pytorch:latest

ARG UID=1000
ARG DOCKERUSER=docker
ARG DOCKERUSERCOMMENT=
ENV DATA_PATH=
ENV EPOCH=1000

RUN useradd -d /${DOCKERUSER} -m \
            -u ${UID} -U \
            -s /usr/bin/bash \
            -c "${DOCKERUSERCOMMENT}" ${DOCKERUSER} && \
    echo "${DOCKERUSER} ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

RUN pip install 'termcolor>=2.4.0' \
                'gdown>=5.1.0' \
                'hydra-core >=1.3.2'\
                'einops >=0.8.0'\
                'pymunk >=6.6.0' \
                'zarr >=2.17.0' \
                'numba >=0.59.0' \
                'opencv-python >=4.9.0' \
                'diffusers >=0.27.2' \
                'torchvision >=0.17.1' \
                'datasets >=2.19.0' \
                'numpy<1.24' \
                matplotlib

USER ${DOCKERUSER}
WORKDIR /${DOCKERUSER}/app

ENTRYPOINT ["sh", "-c", "python ./imitation/compute_stats --path $DATA_PATH && python ./imitation/train_script --path $DATA_PATH --epoch $EPOCH"]