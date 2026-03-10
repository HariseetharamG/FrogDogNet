#!/bin/bash

#cd ..

# custom config
#DATA=/raid/biplab/hari/suraj/RepoAPPLeNet/RSICD
TRAINER=CoCoOp

DATASET=$1
SEED=$2

# Define mapping between dataset argument names and actual folder names
declare -A DATASET_PATHS
DATASET_PATHS["resisc45"]="/apps/hari/datasets/RESISC45"
DATASET_PATHS["patternnet"]="/apps/hari/datasets/PatternNet"
DATASET_PATHS["mlrsnet"]="/apps/hari/datasets/MLRSNet"
DATASET_PATHS["rsicd"]="/apps/hari/datasets/RSICD"

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
SHOTS=16
LOADEP=50
SUB=all

#--load-epoch 50 \

DIR=outputs/crosstransfer/tests/${TRAINER}/${CFG}_shots${SHOTS}/${DATASET}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "The results already exist in ${DIR}"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file yaml/datasets/${DATASET}.yaml \
    --config-file yaml/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir outputs/crosstransfer/patternnet/${TRAINER}/${CFG}_shots${SHOTS}/seed${SEED} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi
