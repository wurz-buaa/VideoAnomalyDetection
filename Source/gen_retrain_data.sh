export CUDA_VISIBLE_DEVICES=0

# parameter setting
batchsize=50
dataset=UCSDped2
#dataset=UCSDped1
#dataset=shanghaitech
#dataset=avenue

modeldir=hvad-32-16-8-release

python3 gen_test.py $dataset 0-3 $modeldir test
python3 gen_normalTest.py $dataset 0-3 $modeldir test
python3 gen_testmask.py $dataset 0-3 $modeldir test