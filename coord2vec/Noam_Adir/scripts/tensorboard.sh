#!/usr/bin/env bash
. ~/.bashrc
conda activate coord2vec
cd '/data/home/morpheus/coord2vec_noam/coord2vec/Noam_Adir'
tensorboard --logdir=tb_logs --host=0.0.0.0 --port=6667
