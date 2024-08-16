
PROJ_ROOT="/root/animation_training/src/video_crafter"                      # root directory for saving experiment logs
EXPNAME="vc_mergedsrunjump_weap_origvc2"          # experiment name
TRAIN_DATADIR="/root/ucf_ds/train"  # dataset directory
VAL_DATADIR="/root/ucf_ds/val"  # dataset directory
DATADIR="/root/ucf_ds"  # dataset directory
#AEPATH="/root/vc2/ae/ae_sky.ckpt"    # pretrained video autoencoder checkpoint

CONFIG="configs/train_t2v_sample_cb_less_val.yaml"
# CONFIG="configs/lvdm_short/sky.yaml"
# OR CONFIG="configs/videoae/ucf.yaml"
# OR CONFIG="configs/videoae/taichi.yaml"
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
#\
#model.params.first_stage_config.params.ckpt_path=$AEPATH

# -------------------------------------------------------------------------------------------------
# commands for multi nodes training
# - use torch.distributed.run to launch main.py
# - set `gpus` and `lightning.trainer.num_nodes`

# For example:

# python -m torch.distributed.run \
#     --nproc_per_node=8 --nnodes=$NHOST --master_addr=$MASTER_ADDR --master_port=1234 --node_rank=$INDEX \
#     main.py \
#     --base $CONFIG \
#     -t --gpus 0,1,2,3,4,5,6,7 \
#     --name $EXPNAME \
#     --logdir $PROJ_ROOT \
#     --auto_resume True \
#     lightning.trainer.num_nodes=$NHOST \
#     data.params.train.params.data_root=$DATADIR \
#     data.params.validation.params.data_root=$DATADIR
