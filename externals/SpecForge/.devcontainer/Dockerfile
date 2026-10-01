FROM lmsysorg/sglang:dev

# Create non-root user with specified UID and GID
# NOTE: Replace with your own UID and GID. This is a workaround from https://github.com/microsoft/vscode-remote-release/issues/49#issuecomment-489060908.
ARG HOST_UID=1003
ARG HOST_GID=1003
RUN groupadd -g $HOST_GID devuser && \
    useradd -m -u $HOST_UID -g $HOST_GID -s /bin/zsh devuser

# Give devuser sudo access
RUN apt-get update && apt-get install -y sudo && \
    echo "devuser ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/devuser && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# Set up oh-my-zsh for devuser
RUN cp -r /root/.oh-my-zsh /path/to/local-resource && \
    cp /root/.zshrc /path/to/local-resource && \
    cp /root/.vimrc /path/to/local-resource && \
    cp /root/.tmux.conf /path/to/local-resource && \
    sed -i 's|/root/.oh-my-zsh|/path/to/local-resource|g' /path/to/local-resource && \
    chown -R devuser:devuser /path/to/local-resource/

# Set workspace directory and ownership
WORKDIR /sgl-workspace/sglang
RUN chown -R devuser:devuser /sgl-workspace

# Switch to devuser
USER devuser

# Install rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
