
PROJ_ROOT="/root/animation_training/src/video_crafter"                      # root directory for saving experiment logs
EXPNAME="vc_mergedspinkick"          # experiment name
TRAIN_DATADIR="/root/ucf_ds/train"  # dataset directory
VAL_DATADIR="/root/ucf_ds/val"  # dataset directory
DATADIR="/root/ucf_ds"  # dataset directory

CONFIG="configs/train_t2v.yaml"
ckpt='/root/vc2/model.ckpt'

# run
export TOKENIZERS_PARALLELISM=false
python train.py \
--base $CONFIG \
-t --gpus 0, \
--name $EXPNAME \
--logdir $PROJ_ROOT \
--auto_resume False \
--load_from_checkpoint $ckpt \
lightning.trainer.num_nodes=1 \
data.params.train.params.data_root=$DATADIR \
data.params.validation.params.data_root=$DATADIR
