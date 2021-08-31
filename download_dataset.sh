# !/usr/bin/bash
# This script downloads the RefCOCO+ dataset and its bottom-up-attention features.

# dataset
echo "Downloading RefCOCO+ dataset ... "
wget -O ./data/referExpression/refcoco+/instances.json https://www.dropbox.com/sh/4jqadcfkai68yoe/AADsJzMR3P5CRqbDN3TPzdG6a/refcoco%2B/instances.json?dl=0
wget -O ./data/referExpression/refcoco+/refs(unc).p https://www.dropbox.com/sh/4jqadcfkai68yoe/AAB7V1BaagaXAvV_jCEgrQOma/refcoco%2B/refs%28unc%29.p?dl=0

# features
echo "Downloading bottom-up features ... "
wget -O ./data/referExpression/refcoco+_gt_resnet101_faster_rcnn_genome.lmdb/data.mdb https://www.dropbox.com/sh/4jqadcfkai68yoe/AAAfUyY6CLTDQjrjBzWpqBkua/refcoco%2B_gt_resnet101_faster_rcnn_genome.lmdb/data.mdb?dl=0
wget -O ./data/referExpression/refcoco+_gt_resnet101_faster_rcnn_genome.lmdb/lock.mdb https://www.dropbox.com/sh/4jqadcfkai68yoe/AAALVeDoHRomCu-AsyUwnM4Fa/refcoco%2B_gt_resnet101_faster_rcnn_genome.lmdb/lock.mdb?dl=0
wget -O ./data/referExpression/refcoco+_resnet101_faster_rcnn_genome.lmdb/data.mdb https://www.dropbox.com/sh/4jqadcfkai68yoe/AABlj40UeJ7qSql-7CkFtxeea/refcoco%2B_resnet101_faster_rcnn_genome.lmdb/data.mdb?dl=0
wget -O ./data/referExpression/refcoco+_resnet101_faster_rcnn_genome.lmdb/lock.mdb https://www.dropbox.com/sh/4jqadcfkai68yoe/AACoa8Wsoe21EkvWS2_hR48ka/refcoco%2B_resnet101_faster_rcnn_genome.lmdb/lock.mdb?dl=0

