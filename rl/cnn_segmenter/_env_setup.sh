reborn_dir=/home/dmnph/reborn-uasr

source ~/.bashrc

conda create -n reborn
conda activate reborn

conda env list

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install pip==24.0

# CUDA_VISIBLE_DEVICES=0 python3 _check_cuda.py

# install fairseq
echo "Install fairseq"
cd ${reborn_dir}/fairseq
pip install --editable .

# install KenLM
echo "Install KenLM"
cd ${reborn_dir}
cd ..
git clone https://github.com/kpu/kenlm
cd kenlm
mkdir -p build
cd build
cmake ..
make -j 4
cd ..
python setup.py install
export KENLM_ROOT=$(pwd)
cd ..

# install other requirements
echo "Install other requirements"
cd ${reborn_dir}
pip install -r requirements.txt
