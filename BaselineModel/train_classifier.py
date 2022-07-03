import argparse
from tqdm import tqdm
import random
import os
import json
from collections import defaultdict

from models.classifier import Classifier

from models.utils import InputExample
from models.utils import load_intent_datasets, load_intent_examples, sample, print_results
from models.utils import calc_in_acc
from models.utils import THRESHOLDS
from sklearn.metrics import f1_score, recall_score, precision_score

import time

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",
                        default=42,
                        type=int,
                        help="Random seed")
    parser.add_argument("--bert_model",
                        default='roberta-base',
                        type=str,
                        help="BERT model")
    parser.add_argument("--train_batch_size",
                        default=15,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=25.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--no_cuda",
                        action='store_true', #Store_true: false
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--max_grad_norm', help='gradient clipping for Max gradient norm.', required=False, default=1.0,
                        type=float)
    parser.add_argument('--label_smoothing',
                        type = float,
                        default = 0.1,
                        help = 'Coefficient for label smoothing (default: 0.1, if 0.0, no label smoothing)')
    parser.add_argument('--max_seq_length',
                        type = int,
                        default = 128,
                        help = 'Maximum number of paraphrases for each sentence')
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lowercase input string")

    # Special params
    parser.add_argument('--train_file_path',
                        type = str,
                        default = None,
                        help = 'Training data path')
    parser.add_argument('--dev_file_path',
                        type = str,
                        default = None,
                        help = 'Validation data path')
    parser.add_argument('--oos_dev_file_path',
                        type = str,
                        default = None,
                        help = 'Out-of-Scope validation data path')

    parser.add_argument('--output_dir',
                        type = str,
                        default = None,
                        help = 'Output file path')
    parser.add_argument('--save_model_path',
                        type=str,
                        default='',
                        help='path to save the model checkpoints')

    parser.add_argument('--few_shot_num',
                        type = int,
                        default = 5,
                        help = 'Number of training examples for each class')
    parser.add_argument('--num_trials',
                        type = int,
                        default = 10,
                        help = 'Number of trials to see robustness')

    parser.add_argument("--do_predict",
                        action='store_true',
                        help="do_predict the model")
    parser.add_argument("--do_final_test",
                        action='store_true',
                        help="do_predict the model")

    args = parser.parse_args()
    random.seed(args.seed)

    N = args.few_shot_num
    T = args.num_trials

    train_file_path = args.train_file_path
    dev_file_path = args.dev_file_path
    train_examples, dev_examples = load_intent_datasets(train_file_path, dev_file_path, args.do_lower_case)
    sampled_tasks = [sample(N, train_examples) for i in range(T)]
    
    if args.oos_dev_file_path is not None:
        oos_dev_examples = load_intent_examples(args.oos_dev_file_path, args.do_lower_case)
    else:
        oos_dev_examples = []
        
    label_lists = []
    intent_train_examples = []
    intent_dev_examples = []
    intent_oos_dev_examples = []

    for i in range(T):
        tasks = sampled_tasks[i]
        label_lists.append([])
        intent_train_examples.append([])
        intent_dev_examples.append([InputExample(e.text, None, e.label) for e in dev_examples])
        intent_oos_dev_examples.append([InputExample(e.text, None, None) for e in oos_dev_examples])

        for task in tasks:
            label = task['task']
            examples = task['examples']
            label_lists[-1].append(label)

            for j in range(len(examples)):
                intent_train_examples[-1].append(InputExample(examples[j], None, label))

    if args.output_dir is not None:
        folder_name = '{}/{}-shot-{}/'.format(args.output_dir, N, args.bert_model)

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        file_name = 'batch_{}---epoch_{}---lr_{}'.format(args.train_batch_size, args.num_train_epochs, args.learning_rate)
        file_name = '{}__oos-threshold'.format(file_name)

        if args.do_final_test:
            file_name = file_name + '_TEST.txt'
        else:
            file_name = file_name + '.txt'

        f = open(folder_name+file_name, 'w')
    else:
        f = None

    for j in range(T):
        save_model_path = '{}_{}'.format(folder_name + args.save_model_path, j + 1)
        if os.path.exists(save_model_path):
            assert args.do_predict
        else:
            assert not args.do_predict

        if args.save_model_path and os.path.exists(save_model_path):
            model = Classifier(path = save_model_path,
                               label_list = label_lists[j],
                               args = args)

        else:
            model = Classifier(path = None,
                               label_list = label_lists[j],
                               args = args)
            
            model.train(intent_train_examples[j])

            if args.save_model_path:
                if not os.path.exists(save_model_path):
                    os.mkdir(save_model_path)
                model.save(save_model_path)

        in_domain_preds = model.evaluate(intent_dev_examples[j])

        in_domain_labels = [el[1] for el in in_domain_preds]

        true_labels = [el.label for el in dev_examples]
        f1_score_val = f1_score(true_labels, in_domain_labels, average='macro')

        in_acc = calc_in_acc(dev_examples, in_domain_preds, THRESHOLDS)
        precision = precision_score(true_labels, in_domain_labels,
                                    average='macro')

        recall = recall_score(true_labels, in_domain_labels,
                              average='macro')

        precision_vals = [precision for _ in range(len(THRESHOLDS))]
        recall_vals = [recall for _ in range(len(THRESHOLDS))]
        f1_score_vals = [f1_score_val for _ in range(len(THRESHOLDS))]


        print_results(THRESHOLDS, in_acc, f1_score_vals, precision_vals,
                      recall_vals)
            
        if f is not None:
            for i in range(len(in_acc)):
                f.write('{},{},{},{} '.format(in_acc[i]))
            f.write('\n')
        
    if f is not None:
        f.close()


if __name__ == '__main__':
    main()
