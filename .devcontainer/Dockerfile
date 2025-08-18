FROM node:lts AS node

FROM python:3.12-slim-bookworm
ARG USERNAME=node

# Set `DEVCONTAINER` environment variable to help with orientation
ENV DEVCONTAINER=true
# Set environment variables for working with Python
# ---
# PYTHONDONTWRITEBYTECODE prevents Python from writing .pyc files
# PYTHONUNBUFFERED ensures that Python output is sent straight to the terminal
# UV_COMPILE_BYTECODE enables bytecode compilation for uvx
# UV_LINK_MODE sets how uvx links Python packages
# UV_SYSTEM_PYTHON ensures uvx uses the system Python interpreter
# ---
# These settings help with performance and compatibility in the container environment
# They also ensure that Python packages are installed in a way that works well with the container's filesystem
# and that the Python environment is consistent across different runs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    sudo \
    less \
    procps \
    sudo \
    fzf \
    zsh \
    man-db \
    unzip \
    gnupg2 \
    gh \
    iptables \
    ipset \
    iproute2 \
    dnsutils \
    aggregate \
    jq \
    libcap2-bin \
    net-tools \
    locales \
    software-properties-common \
    ca-certificates \
    screen \
    netcat-openbsd

# Add github cli
RUN (type -p wget >/dev/null || (apt update && apt install wget -y)) \
	&& mkdir -p -m 755 /etc/apt/keyrings \
	&& out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
	&& cat $out | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
	&& chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
	&& mkdir -p -m 755 /etc/apt/sources.list.d \
	&& echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
	&& apt update \
	&& apt install gh -y  \
  && rm -rf /var/lib/apt/lists/*

# Create node user and add to docker group
RUN groupadd docker || true
RUN useradd -m -s /bin/bash ${USERNAME}
RUN usermod -aG sudo ${USERNAME}
RUN usermod -aG docker ${USERNAME}
RUN echo '${USERNAME} ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

# Ensure default node user has access to /usr/local/share
RUN mkdir -p /usr/local/share/npm-global && \
  chown -R ${USERNAME}:${USERNAME} /usr/local/share

  # Install global packages
ENV NPM_CONFIG_PREFIX=/usr/local/share/npm-global
ENV PATH=$PATH:/usr/local/share/npm-global/bin

# Install Node from the node stage
# Copy the entire Node.js installation to preserve all dependencies and symlinks
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/bin/npm /usr/local/bin/npm
COPY --from=node /usr/local/bin/npx /usr/local/bin/npx
COPY --from=node /usr/local/bin/corepack /usr/local/bin/corepack

# Install uvx for managing Python virtual environments
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Ensure npm/npx symlinks are correct
RUN ln -sf /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm && \
    ln -sf /usr/local/lib/node_modules/npm/bin/npx-cli.js /usr/local/bin/npx && \
    ln -sf /usr/local/lib/node_modules/corepack/dist/corepack.js /usr/local/bin/corepack || true

# Create workspace and config directories and set permissions
RUN mkdir -p /workspace /home/node/.claude && \
  chown -R node:node /workspace /home/node/.claude

# Persist bash history.
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  && mkdir /commandhistory \
  && touch /commandhistory/.bash_history \
  && chown -R $USERNAME /commandhistory

# Use sudo for docker commands
RUN echo "alias docker='sudo docker'" >> /home/${USERNAME}/.bashrc

WORKDIR /workspace

RUN ARCH=$(dpkg --print-architecture) && \
  wget "https://github.com/dandavison/delta/releases/download/0.18.2/git-delta_0.18.2_${ARCH}.deb" && \
  sudo dpkg -i "git-delta_0.18.2_${ARCH}.deb" && \
  rm "git-delta_0.18.2_${ARCH}.deb"

# Set the default shell to zsh rather than sh
ENV SHELL=/bin/zsh

# Default powerline10k theme
RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.0/zsh-in-docker.sh)" -- \
  -p git \
  -p fzf \
  -a "source /usr/share/doc/fzf/examples/key-bindings.zsh" \
  -a "source /usr/share/doc/fzf/examples/completion.zsh" \
  -a "export PROMPT_COMMAND='history -a' && export HISTFILE=/commandhistory/.bash_history" \
  -x


# Install Claude
RUN npm install -g npm@latest && npm install -g @anthropic-ai/claude-code
RUN uvx pip install --upgrade SuperClaude && uvx SuperClaude install --profile developer --yes

USER ${USERNAME}
