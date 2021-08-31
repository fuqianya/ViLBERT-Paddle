"""
eval_utils.py
~~~~~~~~~~~~

This module contains utils for evaluation.
"""

# paddle
import paddle

def eval_rec(batch, model, crit, results):
    """Evaluate the ViLBERT on the referring expression comprehension (rec) task."""
    features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = batch
    batch_size = features.shape[0]

    with paddle.no_grad():
        vil_prediction, vil_logit, vil_binary_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit \
            = model(question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask)

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

    for i in range(select_idx.shape[0]):
        results.append({'id': question_id[i].item(), 'target': select_idx[i].item(), 'IOU': select_target[i].item()})

    return float(loss), float(batch_score), batch_size, results