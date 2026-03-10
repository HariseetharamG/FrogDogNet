#!/bin/bash

#cd ..

# custom config
#DATA=/raid/biplab/hari/suraj/RepoAPPLeNet/RSICDv2
TRAINER=CoCoOp

DATASET=$1
SEED=$2

# Define mapping between dataset argument names and actual folder names
declare -A DATASET_PATHS
DATASET_PATHS["resisc45v2"]="/apps/hari/datasets/RESISC45v2"
DATASET_PATHS["patternnetv2"]="/apps/hari/datasets/PatternNetv2"
DATASET_PATHS["mlrsnetv2"]="/apps/hari/datasets/MLRSNetv2"
DATASET_PATHS["rsicdv2"]="/apps/hari/datasets/RSICDv2"

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

#--load-epoch 20 \

DIR=outputs/domain_generalization/tests/${TRAINER}/${CFG}_shots${SHOTS}/${DATASET}/seed${SEED}
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
    --model-dir outputs/domain_generalization/patternnetv2/${TRAINER}/${CFG}_shots${SHOTS}/seed${SEED} \
    --load-epoch ${LOADEP} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi 
