set -e
echo env of PyToch 2 for TotalSegmentator

CONDA_P=${1-"$HOME/miniconda3"}
CONDA_BIN=$CONDA_P/bin
ENV=py38_pt222
if [ ! -d $CONDA_P/envs/$ENV ]; then
    $CONDA_BIN/conda create --name $ENV python=3.8 -y
fi
ENV_BIN=$CONDA_P/envs/$ENV/bin

# conda install -n $ENV click scipy tqdm ninja matplotlib pandas -y
$ENV_BIN/pip install nibabel # medpy simpleitk itk opencv-python-headless scikit-learn nltk tensorboard numba monai
$ENV_BIN/pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cu118
$ENV_BIN/pip install TotalSegmentator==1.5.7
# conda install -n $ENV jupyter -y # solve requests ver conflict with TotalSegmentator
