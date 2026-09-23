# Base image with Python 3.9
FROM python:3.9-slim-bullseye

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Install system dependencies (you may need to tweak this based on MimicGen's specific requirements)
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    curl \
    ca-certificates \
    libhdf5-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglu1-mesa-dev \
    libglfw3-dev \
    libglew-dev \
    libglib2.0-0 \
    libosmesa6-dev \
    libsm6 \
    libxinerama-dev \
    libxcursor-dev \
    libxi-dev \
    libxrandr-dev \
    libxxf86vm-dev \
    libxrender-dev \
    libxfixes-dev \
    libxext-dev \
    libx11-dev \
    libxkbcommon-x11-0 \
    libxkbcommon-dev \
    libxkbcommon0 \
    libwayland-dev \
    libxcb-xinerama0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-xfixes0 \
    libxcb-shape0 \
    libxcb-randr0 \
    libxcb-sync1 \
    libxcb-xkb1 \
    libx11-xcb1 \
    libxi6 \
    libxtst6 \
    libxrender1 \
    libxcomposite1 \
    libxcursor1 \
    libxdamage1 \
    libxext6 \
    libxfixes3 \
    libxrandr2 \
    libxss1 \
    mesa-utils \
    pkg-config \
    sudo \
    x11-apps \
    x11proto-core-dev \
    && rm -rf /var/lib/apt/lists/*

# Install MuJoCo binaries for Python bindings
ARG MUJOCO_VERSION=3.5.0
ENV MUJOCO_PATH=/opt/mujoco
ENV MUJOCO_PLUGIN_PATH=${MUJOCO_PATH}/plugin
ENV PATH=${MUJOCO_PATH}/bin:${PATH}
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MUJOCO_PATH}/lib
RUN mkdir -p /opt \
    && curl -L -o /tmp/mujoco.tar.gz https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz \
    && tar -xzf /tmp/mujoco.tar.gz -C /opt \
    && mv /opt/mujoco-${MUJOCO_VERSION} ${MUJOCO_PATH} \
    && rm /tmp/mujoco.tar.gz

# Switch to non-root user
WORKDIR /home/user/oplearn
COPY . .
RUN apt-get update && apt-get install -y cmake g++ build-essential
RUN apt-get update && apt-get install -y tmux nano && apt-get clean
# Python setup
RUN pip install --upgrade pip
RUN pip install wheel

RUN mkdir -p /opt/src
RUN git clone --depth 1 https://github.com/helenlu66/robosuite-task-zoo.git /opt/src/robosuite-task-zoo
RUN git clone --depth 1 https://github.com/helenlu66/tarski.git /opt/src/tarski
RUN git clone --depth 1 https://github.com/helenlu66/mimicgen.git /opt/src/mimicgen
RUN git clone --depth 1 https://github.com/helenlu66/robosuite.git /opt/src/robosuite
RUN pip install -e /opt/src/robosuite
RUN pip install -e /opt/src/robosuite-task-zoo
RUN pip install -e /opt/src/mimicgen
RUN pip install -e /opt/src/tarski
RUN pip install -r requirements.txt

