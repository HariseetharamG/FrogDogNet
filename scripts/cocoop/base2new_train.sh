#!/bin/bash

#cd ../..

# custom config


#DATA=/apps/hari/datasets/MLRSNet
TRAINER=CoCoOp
# TRAINER=CoOp

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
# CFG=vit_b16_ctxv1  # uncomment this when TRAINER=CoOp
# CFG=vit_b16_ep50_ctxv1  # uncomment this when TRAINER=CoOp and DATASET=imagenet
SHOTS=16


DIR=output/base2new/train_base/${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file yaml/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES base
fi
