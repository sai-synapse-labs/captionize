#!/bin/bash
# setup_local_db.sh
# This script sets up a local rqlite instance for Captionise miners and validators.
# It uses environment variables defined in a .env file or defaults if not set.

# Load .env if it exists
if [ -f .env ]; then
    echo "Loading .env file..."
    export $(grep -v '^#' .env | xargs)
fi

# Set default values if not set in .env
: ${RQLITE_DATA_DIR:="$HOME/.rqlite_data"}
: ${RQLITE_HTTP_PORT:="4001"}
: ${RQLITE_NODE_PORT:="4002"}

echo "Using RQLITE_DATA_DIR: $RQLITE_DATA_DIR"
echo "Using RQLITE HTTP PORT: $RQLITE_HTTP_PORT"
echo "Using RQLITE NODE PORT: $RQLITE_NODE_PORT"

# Check if rqlited is installed; if not, download and install it.
if ! command -v rqlited &> /dev/null; then
    echo "rqlited not found. Installing rqlite..."
    curl -L https://github.com/rqlite/rqlite/releases/download/v8.36.3/rqlite-v8.36.3-linux-amd64.tar.gz -o rqlite.tar.gz
    tar xvfz rqlite.tar.gz --no-same-owner
    mkdir -p "$HOME/bin"
    cp rqlite-v8.36.3-linux-amd64/rqlite* "$HOME/bin/"
    rm -rf rqlite-v8.36.3-linux-amd64* rqlite.tar.gz

    # Add $HOME/bin to PATH if it's not already in PATH
    if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
        echo 'export PATH=$PATH:$HOME/bin' >> ~/.bashrc
        export PATH="$PATH:$HOME/bin"
        echo "Added $HOME/bin to PATH."
    fi
fi

# Create the data directory if it doesn't exist
mkdir -p "$RQLITE_DATA_DIR"

# Start rqlited in the background, logging to rqlite.log
echo "Starting rqlited..."
nohup rqlited -http-addr "0.0.0.0:$RQLITE_HTTP_PORT" -node-addr "0.0.0.0:$RQLITE_NODE_PORT" "$RQLITE_DATA_DIR" > rqlite.log 2>&1 &
echo "rqlited started. Logs available in rqlite.log"
