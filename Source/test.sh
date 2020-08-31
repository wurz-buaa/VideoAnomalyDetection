export CUDA_VISIBLE_DEVICES=0

# parameter setting
dataset=UCSDped2
#dataset=UCSDped1
#dataset=shanghaitech
#dataset=avenue

modeldir=hvad-32-16-8-release

####################################################################
#echo [5] DETECTION: CALCULATING GENERATED FRAMES OF TESTING VIDEOS

python3 test_compute_recon_brox_v5_largev2_reshape_release.py $dataset 0 0 False $modeldir hvad-gan-layer0-v5-brox
python3 test_compute_recon_brox_v5_largev2_reshape_release.py $dataset 1 0 False $modeldir hvad-gan-layer0-v5-brox
python3 test_compute_recon_brox_v5_largev2_reshape_release.py $dataset 0 3 False $modeldir hvad-gan-layer0-v5-brox
python3 test_compute_recon_brox_v5_largev2_reshape_release.py $dataset 1 3 False $modeldir hvad-gan-layer0-v5-brox

####################################################################
#echo [6] DETECTION: FINDING ANOMALY OBJECTS
# detecting anomalies using features at 0-3 levels

python3 test_hvad_v5_brox_hier_2thesh_release_v4_largev2_reshape_release.py $dataset 0-3 $modeldir test 1 1
#echo end!!