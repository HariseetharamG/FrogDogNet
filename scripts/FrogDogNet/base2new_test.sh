#!/bin/bash


TRAINER=FrogDogNet
# TRAINER=CoOp

DATASET=$1
SEED=$2

# Define mapping between dataset argument names and actual folder names
declare -A DATASET_PATHS
DATASET_PATHS["resisc45"]="/Replace with your dataset location/RESISC45"
DATASET_PATHS["patternnet"]="/Replace with your dataset location/PatternNet"
DATASET_PATHS["mlrsnet"]="/Replace with your dataset location/MLRSNet"
DATASET_PATHS["rsicd"]="/Replace with your dataset location/RSICD"

# Get the correct data path based on the dataset argument
DATA=${DATASET_PATHS[$DATASET]}

# Check if the DATA path was found
if [ -z "$DATA" ]; then
  echo "Error: Dataset '$DATASET' not recognized. Please check the dataset name."
  exit 1
fi

# Print the configuration for verification
echo "Running with DATA=$DATA, TRAINER=$TRAINER, DATASET=$DATASET, SEED=$SEED"

CFG=vit_b16_c4_ep10_batch1_ctxv1
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
SHOTS=16
LOADEP=50
SUB=new


COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi
