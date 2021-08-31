"""
eval.py
~~~~~~~

A script to eval ViLBERT model on Referring Expression Comprehension (REC) task.
"""
import os
import time
import json
import yaml
import random
import logging
import argparse

import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from paddlenlp.transformers.bert.tokenizer import BertTokenizer

# paddle
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader

# model
from model.vilbert import BertConfig
from model.vilbert import VILBertForVLTasks
# dataset
from model.rec_dataset import ReferExpressionDataset
# image feature reader
from utils.io import ImageFeaturesH5Reader
# eval utils
from utils.eval_utils import eval_rec

LossMap = {
    'BCEWithLogitLoss': nn.BCEWithLogitsLoss(reduction='mean'),
    'CrossEntropyLoss': nn.CrossEntropyLoss()
}

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def main(args):
    # task config
    with open('vlbert_tasks.yml', 'r') as f:
        task_cfg = edict(yaml.safe_load(f))

    # task name
    task_id = 'TASK4'  # REC
    task_cfg = task_cfg[task_id]

    # set random seed
    random.seed(args.seed)
    paddle.seed(args.seed)
    np.random.seed(args.seed)

    # save path
    timeStamp = args.from_pretrained.split('/')[1]
    savePath = os.path.join(args.output_dir, timeStamp)
    if not os.path.exists(savePath): os.makedirs(savePath)

    config = BertConfig.from_json_file(args.config_file)
    bert_weight_name = json.load(open("config/" + args.bert_model + "_weight_name.json", "r"))

    logger.info("Loading %s Dataset with batch size %d" % (task_cfg['name'], args.batch_size))

    # init tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # init image features reader
    image_features_reader = ImageFeaturesH5Reader(task_cfg['features_h5path1'])
    gt_image_features_reader = ImageFeaturesH5Reader(task_cfg['features_h5path2'])

    # load dataset
    rec_dataset = ReferExpressionDataset(task=task_cfg['name'],
                                         split=task_cfg['val_split'],
                                         dataroot=task_cfg['dataroot'],
                                         image_features_reader=image_features_reader,
                                         gt_image_features_reader=gt_image_features_reader,
                                         annotations_jsonpath=task_cfg['val_annotations_jsonpath'],
                                         tokenizer=tokenizer,
                                         padding_index=0,
                                         max_seq_length=task_cfg['max_seq_length'],
                                         max_region_num=task_cfg['max_region_num'])

    # set up dataloader
    rec_dataloader = DataLoader(dataset=rec_dataset,
                                shuffle=False,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

    # set up model
    model = VILBertForVLTasks(config, num_labels=rec_dataset.num_labels)
    checkpoint = paddle.load(args.from_pretrained)
    model.set_state_dict(checkpoint)

    # set up criterion
    crit = LossMap[task_cfg['loss']]

    print("***** Running Evaluation *****")
    print("  Num Iters: ", len(rec_dataloader))
    print("  Batch size: ", args.batch_size)

    logger.info('Start to eval ... ')
    model.eval()
    results = []
    time.sleep(0.1)

    num_correct, num_total = 0., 0.
    for batch in tqdm(rec_dataloader):
        loss, score, batch_size, results = eval_rec(batch, model, crit, results)
        num_total += batch_size
        num_correct += score

    print('acc: ', num_correct / num_total)
    json_path = os.path.join(savePath, task_cfg['val_split'])
    json.dump(results, open(json_path + '_result.json', 'w'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--bert_model', default='bert-base-uncased', type=str,
                        help='Bert pre-trained model selected in the list: bert-base-uncased'
                             'bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.')

    parser.add_argument("--from_pretrained", type=str, help="dir of the pretrained model.",
                        default="checkpoints/refcoco+_bert_base_6layer_6conect-pretrained/paddle_model_19.pdparams")

    parser.add_argument("--output_dir", default="results", type=str,
                        help="The output directory where the model checkpoints will be written.",)

    parser.add_argument("--config_file", default="config/bert_base_6layer_6conect.json", type=str,
                        help="The config file which specified the model details.")

    parser.add_argument("--do_lower_case", default=True, type=bool,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")

    parser.add_argument("--seed", type=int, default=42, help="rando                                                                                                              m seed for initialization")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers in the dataloader.")
    parser.add_argument("--batch_size", default=512, type=int, help="what is the batch size?")

    args = parser.parse_args()

    # call main()
    main(args)