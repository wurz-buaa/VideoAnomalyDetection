export CUDA_VISIBLE_DEVICES=0

# parameter setting
batchsize=50
dataset=UCSDped2
#dataset=UCSDped1
#dataset=shanghaitech
#dataset=avenue


####################################################################
#echo [1] EXTRACTING FEATURES

## extracting motion feature
python3 feat_optical_flow_extract.py $dataset 0

## extracting raw feature data
#resz=[240,360]
#resz=[158,238]
resz=[256,256]
python3 feat_raw_extract.py $dataset $resz 0

####################################################################
#echo [2] TRAINING DENOISING AUTOENCODERS
# Denoising Autoencoder architecture
encoder=32-16-8
modeldir=hvad-32-16-8-release

# for our experiments
disp_freq=10
save_freq=10
num_epochs=500
device='/device:GPU:0'

python3 train_hvad_CAEv5_brox_release.py 0 $dataset $batchsize $encoder $disp_freq $save_freq $num_epochs $device
python3 train_hvad_CAEv5_brox_release.py 1 $dataset $batchsize $encoder $disp_freq $save_freq $num_epochs $device

####################################################################
#echo [3] EXTRACT HIGH-LEVEL REPRESENTATION FEATURES
python3 extract_high_feat_from_cae_brox_batch_v5_release.py $dataset 1 $batchsize $modeldir all $device
python3 extract_high_feat_from_cae_brox_batch_v5_release.py $dataset 2 $batchsize $modeldir all $device

####################################################################
#echo [4] TRAINING CONDITIONAL GANS

##################### skip_frame:Ped1\Ped2=1, avenue\shanghaitech=2
skip_frame=1
python3 train_hvad_GANv5_brox_largev2_reshape_release.py $dataset train 0 False AtoB $skip_frame $modeldir 0
python3 train_hvad_GANv5_brox_largev2_reshape_release.py $dataset train 0 False BtoA $skip_frame $modeldir 0
python3 train_hvad_GANv5_brox_largev2_reshape_release.py $dataset train 3 False AtoB $skip_frame $modeldir 0
python3 train_hvad_GANv5_brox_largev2_reshape_release.py $dataset train 3 False BtoA $skip_frame $modeldir 0


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