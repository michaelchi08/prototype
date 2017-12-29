#!/bin/bash
set -e
REPO_NAME="prototype"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

setup_virtualenv() {
	echo "Setting virtualenv for ${1}"
	export PYTHONPATH=/usr/local/lib/$1/dist-packages
	virtualenv -p python "$2"
	source $2/bin/activate
}

clear_virtualenv() {
	echo "Clearing virtualenv for ${1}"
	deactivate
	rm -rf $1
}

# Setup environment
sudo bash $DIR/min_env.bash
sudo bash $DIR/py2_env.bash
sudo bash $DIR/py3_env.bash

# Install dependencies
bash $DIR/kitti_raw_data.bash
bash $DIR/../install/apriltags_michigan_install.bash*

# Python 2.7
PY_VER="python"
PY_ENV=${REPO_NAME}_${PY_VER}_env

setup_virtualenv $PY_VER ${PY_ENV}
$PY_VER setup.py install
$PY_VER -m unittest discover
clear_virtualenv ${PY_ENV}

# Python 3.5
PY_VER="python3.5"
PY_ENV=${REPO_NAME}_${PY_VER}_env

setup_virtualenv $PY_VER ${PY_ENV}
$PY_VER setup.py install > /dev/null 2>&1
$PY_VER -m unittest discover
clear_virtualenv ${PY_ENV}
