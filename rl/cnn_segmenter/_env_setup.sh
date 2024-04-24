source ~/.bashrc
conda activate wav2vecu

conda env list

reborn_dir=/home/dmnph/reborn-uasr


# # install fairseq
# echo "Install fairseq"

# cd ${reborn_dir}/fairseq
# pip install --editable .


# # install other requirements
# echo "Install other requirements"
# pip install -r ${reborn_dir}/requirements.txt


# # install KenLM
# echo "Install KenLM"
# cd ${reborn_dir}
# cd ..
# # git clone https://github.com/kpu/kenlm
# cd kenlm
# mkdir -p build
# cd build
# cmake ..
# make -j 4
# cd ..
# python setup.py install
# export KENLM_ROOT=$(pwd)
# cd ..

# # install sklearn
# pip install -U scikit-learn

# # install psutil
# conda install -c conda-forge psutil

# install faiss
conda install -c pytorch faiss-gpu