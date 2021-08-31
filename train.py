"""
train.py
~~~~~~~

A script to finetune ViLBERT model on Referring Expression Comprehension (REC) task.
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
# lr scheduler
from model.optimization import ConstDecayWithWarmup
# dataset
from model.rec_dataset import ReferExpressionDataset
# image feature reader
from utils.io import ImageFeaturesH5Reader

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

def lossFun(batch, model, crit):
    """Fintune the ViLBERT on the referring expression comprehension (rec) task."""
    features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = batch
    batch_size = features.shape[0]

    # get the model output
    vil_prediction, vil_logit, vil_binary_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit = \
            model(question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask)


    loss = crit(vision_logit, target)
    loss = loss.mean() * target.shape[1]
    select_idx = paddle.argmax(vision_logit, axis=1)

    # prepare data for paddle.gather_nd
    gather_index = paddle.zeros(shape=(batch_size, 2), dtype='int64')
    gather_index[:, 0] = paddle.arange(batch_size)
    gather_index[:, 1] = select_idx.reshape((-1, ))
    # apply paddle.gather_nd to gather the target
    select_target = paddle.gather_nd(target.squeeze(2), gather_index)
    batch_score = paddle.sum((select_target > 0.5).cast('float32')).item()

    return loss, batch_score

def eval(batch, model, crit):
    """Evaluate the ViLBERT on the referring expression comprehension (rec) task."""
    features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = batch
    batch_size = features.shape[0]

    with paddle.no_grad():
        vil_prediction, vil_logit, vil_binary_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit = \
                                                model(question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask)

    loss = crit(vision_logit, target)
    loss = loss.mean() * target.shape[1]
    select_idx = paddle.argmax(vision_logit, axis=1)

    # prepare data for paddle.gather_nd
    gather_index = paddle.zeros(shape=(batch_size, 2), dtype='int64')
    gather_index[:, 0] = paddle.arange(batch_size)
    gather_index[:, 1] = select_idx.reshape((-1, ))
    # apply paddle.gather_nd to gather the target
    select_target = paddle.gather_nd(target.squeeze(2), gather_index)
    batch_score = paddle.sum((select_target > 0.5).cast('float32')).item()

    return float(loss), float(batch_score), batch_size

def main(args):
    with open('vlbert_tasks.yml', 'r') as f:
        task_cfg = edict(yaml.load(f))

    # task name
    task_id = 'TASK4'  # REC
    task_cfg = task_cfg[task_id]

    if args.save_name:
        prefix = '-' + args.save_name
    else:
        prefix = ''

    # save path
    timeStamp = args.config_file.split('/')[1].split('.')[0] + prefix
    savePath = os.path.join(args.output_dir, timeStamp)
    if not os.path.exists(savePath): os.makedirs(savePath)
    if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)

    config = BertConfig.from_json_file(args.config_file)
    bert_weight_name = json.load(open("config/" + args.bert_model + "_weight_name.json", "r"))

    # save all the hidden parameters.
    with open(os.path.join(savePath, 'command.txt'), 'w') as f:
        print(args, file=f)  # Python 3.x
        print('\n', file=f)
        print(config, file=f)

    batch_size = args.batch_size // args.gradient_accumulation_steps
    logger.info("Loading %s Dataset with batch size %d" % (task_cfg['name'], batch_size))

    # init tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # init image features reader
    image_features_reader = ImageFeaturesH5Reader(task_cfg['features_h5path1'])
    gt_image_features_reader = ImageFeaturesH5Reader(task_cfg['features_h5path2'])

    # load dataset
    train_dataset = ReferExpressionDataset(task=task_cfg['name'],
                                           split=task_cfg['train_split'],
                                           dataroot=task_cfg['dataroot'],
                                           image_features_reader=image_features_reader,
                                           gt_image_features_reader=gt_image_features_reader,
                                           annotations_jsonpath=task_cfg['val_annotations_jsonpath'],
                                           tokenizer=tokenizer,
                                           padding_index=0,
                                           max_seq_length=task_cfg['max_seq_length'],
                                           max_region_num=task_cfg['max_region_num'])

    val_dataset = ReferExpressionDataset(task=task_cfg['name'],
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
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers)

    val_dataloader = DataLoader(dataset=val_dataset,
                                shuffle=False,
                                batch_size=batch_size,
                                num_workers=args.num_workers)

    # set up training info
    num_iter = len(train_dataloader)
    num_train_optimization_steps = num_iter * args.num_train_epochs // args.gradient_accumulation_steps

    # set up model
    model = VILBertForVLTasks(config, num_labels=train_dataset.num_labels)
    checkpoint = paddle.load(args.from_pretrained)
    model.set_state_dict(checkpoint)

    # set up criterion
    crit = LossMap[task_cfg['loss']]

    no_finetune = ['vil_prediction', 'vil_logit', 'vision_logit', 'linguistic_logit']
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # reduce lr on this epoch
    lr_reduce_list = [12, 16]
    decay_steps = [(decay_epoch / float(args.num_train_epochs) * num_train_optimization_steps)
                   for decay_epoch in lr_reduce_list]

    # set up optimizer
    # train additional classifiers of downstream tasks from sratch
    from_scratch_scheduler = ConstDecayWithWarmup(1e-4, args.warmup_proportion, decay_steps, num_train_optimization_steps)
    from_scratch_optimizer = paddle.optimizer.AdamW(
        learning_rate=from_scratch_scheduler,
        parameters=[p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_finetune)],
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x:x in [
            p.name for n, p in model.named_parameters()
            if any(nd in n for nd in no_finetune)
               and any(nd in n for nd in no_decay)
        ],
        grad_clip=paddle.fluid.clip.ClipGradByValue(1.0)
    )

    # fintune pretrained ViLBERT with slow learning rate
    finetune_scheduler = ConstDecayWithWarmup(args.learning_rate, args.warmup_proportion, decay_steps, num_train_optimization_steps)
    finetune_optimizer = paddle.optimizer.AdamW(
        learning_rate=finetune_scheduler,
        parameters=[p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_finetune)],
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x:x in [
            p.name for n, p in model.named_parameters()
            if not any(nd in n for nd in no_finetune)
               and any(nd in n for nd in no_decay)
        ],
        grad_clip=paddle.fluid.clip.ClipGradByValue(1.0)
    )

    print("***** Running training *****")
    print("  Num Iters: ", num_iter)
    print("  Batch size: ", batch_size)
    print("  Num steps: %d" % num_train_optimization_steps)

    for epoch in range(args.num_train_epochs):
        # compute training loss
        epoch_loss = 0.
        # compute val acc
        epoch_correct, epoch_total = 0., 0.
        print('====>start epoch {}:'.format(epoch))
        time.sleep(0.1)

        # set up mode for training
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc='Training')):
            # forward
            loss, score = lossFun(batch, model, crit)
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # backward
            loss.backward()
            epoch_loss += float(loss)

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # update the schedulers
                finetune_scheduler.step()
                from_scratch_scheduler.step()

                # update parameters
                finetune_optimizer.step()
                from_scratch_optimizer.step()

                # clear grad
                finetune_optimizer.clear_grad()
                from_scratch_optimizer.clear_grad()

        time.sleep(0.5)
        # set up mode for eval
        model.eval()
        for step, batch in enumerate(tqdm(val_dataloader, desc='Eval')):
            loss, score, batch_size = eval(batch, model, crit)
            epoch_correct += score
            epoch_total += batch_size

        # logger training and val info
        logger.info('** ** Epoch {%d} done! Traing loss: %.5f, Val accuracy: %.4f'
                    % (epoch, epoch_loss / num_iter, epoch_correct/epoch_total))

        # Save a trained model
        logger.info("** ** * Saving fine - tuned model on " + timeStamp + "** ** * ")
        output_model_file = os.path.join(savePath, "paddle_model_" + str(epoch) + ".pdparams")
        paddle.save(model.state_dict(), output_model_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--bert_model', default='bert-base-uncased', type=str,
                        help='Bert pre-trained model selected in the list: bert-base-uncased'
                             'bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.')

    parser.add_argument("--from_pretrained", type=str, help="dir of the pretrained model.",
                        default="checkpoints/bert_base_6_layer_6_connect_freeze_0/paddle_model_8.pdparams")

    parser.add_argument("--output_dir", default="checkpoints", type=str,
                        help="The output directory where the model checkpoints will be written.",)

    parser.add_argument("--config_file", default="config/bert_base_6layer_6conect.json", type=str,
                        help="The config file which specified the model details.")

    parser.add_argument('--batch_size', default=256, type=int, help='batch size for training')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay of the parameters')
    parser.add_argument('--decay_factor', default=0.1, type=float, help='decay factor of the learning rate')
    parser.add_argument("--learning_rate", default=4e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=20, type=int, help="Total number of training epochs to perform.")

    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")

    parser.add_argument("--do_lower_case", default=True, type=bool,
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")

    parser.add_argument("--num_workers", type=int, default=12, help="Number of workers in the dataloader.")
    parser.add_argument("--save_name", default='refcoco+', type=str, help="save name for training.")

    parser.add_argument("--optimizer", default='BertAdam', type=str, help="which optimizer used to optimization.")

    args = parser.parse_args()

    # call main
    main(args)
