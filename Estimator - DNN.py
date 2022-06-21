#!/usr/bin/env python
# -*- coding: utf-8 -*-
# *******************************************************************************

# library--------------------------------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf
import data
import pandas as pd
import csv


# Global variable-------------------------------------------------------------------------------------------------------
seq = 12  # used by data
batch_num = 500
step_num = 23000  # 训练总次数
hidden_layer_1 = 20   # 隐藏层1的数量
hidden_layer_2 = 20  # 隐藏层2的数量
hidden_layer_3 = 20   # 隐藏层3的数量
class_num = 2   # 分类数量

if seq == 12:
    sample_start = 1
else:
    sample_start = 32 * (seq - 1) + 1

# probability_csv = "probability_all.csv"
# predict_csv = "TotalPredict.csv"
# model_path = data_path + "model\\hiddenlayer" + str(hidden_layer_1) \
#              + "_" + str(hidden_layer_2) + "_" + str(hidden_layer_3) + \
#              "\\step" + str(step_num) + "\\totalsyntrain" + str(seq)
probability_csv = "probability" + str(seq) + ".csv"
predict_csv = "predict" + str(seq) + ".csv"
data_path = "G:\\workspace_python\\DNN\\Estimator\\data\\"
model_path = data_path + "model\\hiddenlayer" + str(hidden_layer_1) \
             + "_" + str(hidden_layer_2) + "_" + str(hidden_layer_3) + \
             "\\step" + str(step_num) + "\\loop" + str(seq)

print("\nmodel 存储路径是：%s \n预测结果存储文件是: %s\n预测数据文件是: %s"
      % (model_path, probability_csv, predict_csv))

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=batch_num, type=int, help='batch size')
parser.add_argument('--train_steps', default=step_num, type=int, help='number of training steps')


def my_model(features, labels, mode, params):
    """DNN with three hidden layers and learning_rate=0.01."""
    # Create three fully connected layers.

    net = tf.feature_column.input_layer(features, params['feature_columns'])  # Input Layer
    for units in params['hidden_units']:  # Hidden Layer
        net = tf.layers.dense(net,
                              units=units,
                              activation=tf.nn.relu)  # activation = relu

    # Compute logits (1 per class).
    logits = tf.layers.dense(net,
                             params['n_classes'],
                             activation=None)  # Output Layer

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)  # 预测的结果中最大值即种类
    if mode == tf.estimator.ModeKeys.PREDICT:
        print(("*" * 50) + '\nStart predicting...\n' + ("*" * 50))  # 提示启动预测
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],  # 拼成列表[[0],[1]]格式
            'probabilities': tf.nn.softmax(logits),  # 把logits规则化到0~1范围,表示概率
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode,
                                          predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels,
                                                  logits=logits)  # 计算损失率
    print('\nLoss loaded\n')

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')  # 计算精度
    metrics = {'accuracy': accuracy}  # 返回格式
    tf.summary.scalar('accuracy', accuracy[1])  # 为后面图表统计使用

    if mode == tf.estimator.ModeKeys.EVAL:
        print(("*" * 50) + "\nStart evaluating...\n" + ("*" * 50))  # 提示启动预测

        return tf.estimator.EstimatorSpec(
            mode,
            loss=loss,
            eval_metric_ops=metrics)  # 应该在这里修改evaluation的准确率

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN
    print(("*" * 50) + "\nStart training...\n" + ("*" * 50))  # 提示启动训练
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.01)  # optimizer=AdaGrad，learning rate = 0.01
    train_op = optimizer.minimize(loss,
                                  global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode,
                                      loss=loss,
                                      predictions=predicted_classes,
                                      train_op=train_op,
                                      eval_metric_ops={'accuracy': accuracy})


def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    print(("*"*50) + "\nStart main\n" + ("*"*50))
    (train_x, train_y), (test_x, test_y) = data.load_data()

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 3 hidden layer DNN with 10, 10, 10 units respectively.
    classifier = tf.estimator.Estimator(  # custom estimator
        model_fn=my_model,
        model_dir=model_path,
        params={
            'feature_columns': my_feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [hidden_layer_1, hidden_layer_2, hidden_layer_3],
            # The model must choose between 3 classes.
            'n_classes': class_num,
        })
    # classifier = tf.estimator.add_metrics(classifier, my_auc)

    # Train the Model.
    classifier.train(
        input_fn=lambda: data.train_input_fn(train_x, train_y, args.batch_size),
        steps=args.train_steps)

    # Evaluate the model.
    print(("*"*50)+"\nStart evaluating\n"+("*"*50))  # 提示模型函数启动
    eval_result = classifier.evaluate(
        input_fn=lambda: data.eval_input_fn(test_x, test_y, args.batch_size),
    )
    print("\nEvaluation log saved in:\n%s" % classifier.eval_dir(name=None))
    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    data_predict = pd.read_csv(data_path+predict_csv, header=None)
    print("\nPrediction data below:\n%s" % data_predict)
    expected = range(sample_start, sample_start+len(data_predict))
    predict_x = {
        'Ag': list(data_predict[0]),
        'As': list(data_predict[1]),
        'Ba': list(data_predict[2]),
        'Cd': list(data_predict[3]),
        'Cu': list(data_predict[4]),
        'Hf': list(data_predict[5]),
        'Pb': list(data_predict[6]),
        'Rb': list(data_predict[7]),
        'S': list(data_predict[8]),
        'Sb': list(data_predict[9]),
        'Sr': list(data_predict[10]),
        'Te': list(data_predict[11]),
        'W': list(data_predict[12]),
        'Zn': list(data_predict[13]),
        'Zr': list(data_predict[14]),
        'SiO2': list(data_predict[15]),
        'Al2O3': list(data_predict[16]),
        'TFe2O3': list(data_predict[17]),
        'MgO': list(data_predict[18]),
        'CaO': list(data_predict[19]),
        'Na2O': list(data_predict[20]),
        'K2O': list(data_predict[21]),
        'Pth': list(data_predict[22]),
        'Qpx': list(data_predict[23]),
        'Qbg': list(data_predict[24]),
        'Fault': list(data_predict[25])
    }

    predictions = classifier.predict(
        input_fn=lambda: data.eval_input_fn(predict_x,
                                                      labels=None,
                                                      batch_size=args.batch_size))
    prediction_dir = model_path + "\\" + probability_csv
    print("\nProbability result saved in:\n%s\n" % prediction_dir)
    with open(prediction_dir, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sample', 'Prediction Specie', 'Specie_ID', 'Probability'])
    csvfile.close()

    for pred_dict, expec in zip(predictions, expected):
        template = '\nPrediction is "{}" ({:.1f}%), Sample No. "{}"'

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(data.SPECIES[class_id],
                              100 * probability, expec))

        with open(prediction_dir, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([str(expec),
                             str(data.SPECIES[class_id]),
                             str(class_id),
                             str(probability)])
        csvfile.close()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
    print("\nRound %d finished!\n" % seq)
