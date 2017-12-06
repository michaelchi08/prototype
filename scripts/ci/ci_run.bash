#!/bin/bash
REPO_NAME="prototype"
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# # Install dependencies
# bash $DIR/kitti_raw_data.bash
# bash $DIR/../install/apriltags_michigan_install.bash*


# Python 2.7
PY_VER="python"
PY_ENV=${REPO_NAME}_${PY_VER}_env

export PYTHONPATH=/usr/local/lib/$PY_VER/dist-packages
virtualenv -p python "${PY_ENV}"
source ${PY_ENV}/bin/activate

$PY_VER setup.py install > /dev/null 2>&1
$PY_VER -m unittest discover > /dev/null 2>&1

if [ $? -eq 0 ]; then
	echo "PASS!";
else
	echo "FAILED!";
fi

deactivate
rm -rf "${PY_ENV}"


# Python 3.5
PY_VER="python3.5"
PY_ENV=${REPO_NAME}_${PY_VER}_env

export PYTHONPATH=/usr/local/lib/$PY_VER/dist-packages
virtualenv -p python "${PY_ENV}"
source ${PY_ENV}/bin/activate

$PY_VER setup.py install > /dev/null 2>&1
$PY_VER -m unittest discover

if [ $? -eq 0 ]; then
	echo "PASS!";
else
	echo "FAILED!";
fi

deactivate
rm -rf "${PY_ENV}"
