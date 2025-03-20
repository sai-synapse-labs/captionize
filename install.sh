#!/bin/bash


# Check for CUDA availability
if command -v nvidia-smi &> /dev/null; then
    CUDA_AVAILABLE=true
else
    CUDA_AVAILABLE=false
    echo "❌ You are creating an environment without a CUDA compatible GPU. CUDA is a requirement. Not installing captionize."
    exit 1
fi

# ensure pip is installed
python3.10 -m ensurepip

# Install auxiliary packages
sudo apt-get update
sudo apt install python3.10-venv libsndfile1 npm -y
sudo npm install -g pm2 -y
chmod +x setup_local_db.sh
./setup_local_db.sh