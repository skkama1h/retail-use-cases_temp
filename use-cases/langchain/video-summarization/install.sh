#!/bin/bash

# Install Conda
source activate-conda.sh

# one-time installs
if [ "$1" == "--skip" ]
then
	echo "Skipping dependencies"
	activate_conda
else
	echo "Installing dependencies"
	sudo apt update
	sudo apt install -y ffmpeg wget

    # Install Intel Client GPU. Install the Intel graphics GPG public key
    wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
    sudo gpg --yes --dearmor --output /usr/share/keyrings/intel-graphics.gpg

    # Continue installing Client GPU based on Ubuntu OS version
    OS_VER=$(lsb_release -sr | cut -d'.' -f1)
    echo $OS_VER
    if [[ $OS_VER -le 24 ]]; then

        # Configure the repositories.intel.com package repository
        echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu noble client" | \
        sudo tee /etc/apt/sources.list.d/intel-gpu-noble.list

        # Update the package repository meta-data
        sudo apt update

        # Install the compute-related packages
        apt-get install -y libze-intel-gpu1 libze1 intel-opencl-icd clinfo intel-gsc

        # Install PyTorch dependencies
        apt-get install -y libze-dev intel-ocloc
        
    elif [[ $OS_VER -le 22 ]]; then

        # Configure the repositories.intel.com package repository
        echo "deb [arch=amd64,i386 signed-by=/usr/share/keyrings/intel-graphics.gpg] https://repositories.intel.com/gpu/ubuntu jammy client" | \
        sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list

        # Update the package repository meta-data
        sudo apt update

        # Install the compute-related packages
        apt-get install -y libze-intel-gpu1 libze1 intel-opencl-icd clinfo

        # Install PyTorch dependencies    
        apt-get install -y libze-dev intel-ocloc
        
    else
        echo "Only Ubuntu 24.04 and 22.04 supported. Canceling..."
        exit 1
    fi

	CUR_DIR=`pwd`
        cd /tmp
	miniforge_script=Miniforge3-$(uname)-$(uname -m).sh
	[ -e $miniforge_script ] && rm $miniforge_script
	wget "https://github.com/conda-forge/miniforge/releases/latest/download/$miniforge_script"
	bash $miniforge_script -b -u
	# used to activate conda install
	activate_conda
	conda init
	cd $CUR_DIR
fi

# Create python enviornment
conda create -n ovlangvidsumm python=3.10 -y
conda activate ovlangvidsumm
echo 'y' | conda install pip

pip install optimum-intel@git+https://github.com/huggingface/optimum-intel.git nncf openvino-genai timm einops decord
git clone https://github.com/gsilva2016/langchain.git
pushd langchain; git checkout openvino_tts_tool; popd
pip install -e langchain/libs/community
