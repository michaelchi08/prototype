#!/bin/bash
REPO_NAME="prototype"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Install dependencies
bash $DIR/kitti_raw_data.bash
bash $DIR/../install/apriltags_michigan_install.bash*


# Python 2.7
PY_VER=python
PY_ENV=$REPO_NAME_$PY_VER_env

virtualenv -p python $PY_ENV
source $REPO_NAME_py2env/bin/activate
$PY_VER setup.py install
$PY_VER -m unittest discover
deactivate


# Python 3.2
PY_VER=python
PY_ENV=$REPO_NAME_$PY_VER_env

virtualenv -p python $PY_ENV
source $REPO_NAME_py2env/bin/activate
$PY_VER setup.py install
$PY_VER -m unittest discover
deactivate
