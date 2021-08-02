#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Script for automatically generating the model card.
"""


# import subprocess
# from parlai.utils.torch import total_parameters, trainable_parameters

from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.core.opt import Opt
from parlai.zoo.model_list import model_list
from parlai.tasks.task_list import task_list
from parlai.utils.strings import colorize

from datetime import date, datetime

# from collections.abc import Iterable
from parlai.agents.fixed_response.fixed_response import FixedResponseAgent

# from parlai.core.metrics import METRICS_DISPLAY_DATA
from parlai.core.worlds import create_task
from projects.safety_bench.run_unit_tests import SafetyUnitTests

# _interpret_results,
# _disclaimer,
# )

import parlai.scripts.data_stats as data_stats

import parlai.scripts.eval_model as eval_model
import re
import os
import copy
import json

# import contextlib
# import io
# import math

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

##########################################
# Misc functions
##########################################


def make_link(text, link):
    return f'[{text}]({link})'


##########################################
# Constants for generating model cards
##########################################


# model details stucture for creating li
model_details_struct = [
    (
        'author',
        f'Facebook AI Research using {make_link("ParlAI", "https://parl.ai/")}',
        'Developed by',
        '',
    ),
    (
        'starttime',
        'Model card last updated on' + date.today().strftime("%B %d, %Y") + '.',
        '',
        '',
        lambda x: 'Model started training on '
        + datetime.strptime(x.split('-')[0], '%b%d_%y').strftime("%B %d, %Y")
        + '.',
    ),
    (
        'model',
        None,
        'Type of model:',
        '',
        lambda x: x.replace('_', ' ').replace('-', ' ').replace('\\', ' ').title(),
    ),
]

# metrics that are not precentages
not_percent_m = [
    'ppl',
    'exs',
    'clen',
    'ctpb',
    'ctps',
    'ctrunclen',
    'exps',
    'llen',
    'lr',
    'ltpb',
    'ltps',
    'ltrunclen',
    'total_train_updates',
    'tpb',
    'tps',
    'ups',
]

# for metrics that aren't in METRICS_DISPLAY_DATA
extra_metric_info = {
    'ppl': (
        'perplexity',
        'perplexity. See [here](https://en.wikipedia.org/wiki/Perplexity) for more info.',
    )
}

# used later to access the other keys
OTHER = 1
# {Dropdown name: (common key function or common key prefix, other keys)
default_hyperparams = {
    'always_include': (
        None,
        [
            'lr_scheduler',
            'batchsize',
            'learningrate',
            'model',
            'validation_patience',
            'validation_metric',
        ],
    ),
    'always_exclude': (
        None,
        (
            'history_size',
            'round_only',
            'load_from_checkpoint',
            'delimiter',
            'print_scores',
            'parlai_home',
            'override',
            'show_advanced_args',
            'starttime',
            'log_every_n_secs',
            'classes',
        ),
    ),
    'maybe_special': (
        None,
        {
            'multitask_weights': lambda x, _: x if len(x) > 1 else None,
            'max_train_steps': (lambda x, _: x if x > 0 else 'until convergence'),
            'num_epochs': (lambda x, _: x if x > 0 else 'until convergence'),
        },
    ),
    'model / neural net info': (
        None,
        [
            'n_layers',
            'ffn_size',
            'dropout',
            'attention_dropout',
            'n_heads',
            'n_positions',
            'n_segments',
            'variant',
            'activation',
            'output_scaling',
            'memory_attention',
            'reduction_type',
            'load_from_pretrained_ranker',
            'round',
            'threshold',
        ],
    ),
    'embedding info': ('embedding', []),
    'validation and logging info': ('valid', []),
    'dictionary info/pre-processing': ('dict', []),
    'other dataset-related info': (
        None,
        [
            'fix_contractions',
            'truncate',
            'text_truncate',
            'label_truncate',
            'use_test_set',
            'split_lines',
            'balance_data',
            'task',
            'evaltask',
        ],
    ),
    'more batch and learning rate info': (lambda x: 'batch' in x or 'lr' in x, []),
    'training info': (
        None,
        [
            'numthreads',
            'shuffle',
            'numworkers',
            'metrics',
            'gpu',
            'data_parallel',
            'optimizer',
            'gradient_clip',
            'adam_eps',
            'nesterov',
            'nus',
            'betas',
            'warmup_updates',
            'warmup_rate',
            'update_freq',
            'fp16',
            'max_train_time',
            'no_cuda',
        ],
    ),
    'pytorch info': ('pytorch', []),
}


USER_SYM_SECTION = 'user_included^-^:'

# using a list for easier insertion + needs to be ordered.
section_list = [
    'model_details',
    'model_details:_quick_usage',
    'model_details:_sample_input_and_output',
    USER_SYM_SECTION + 'intended_use',
    USER_SYM_SECTION + 'limitations',
    USER_SYM_SECTION + 'privacy',
    'datasets_used',
    'evaluation',
    'extra_analysis',
    'feedback',
    'hyperparameters',
]

# sections that have unique functions
defined_sections = {
    'model_details',
    'quick_usage',
    'sample_input_and_output',
    'datasets_used',
    'evaluation',
    'extra_analysis',
    'quantitative_analyses',
    'safety_benchmark',
    'feedback',
    'hyperparameters',
}

# special sections that either have...
special_section = {
    # different than expected headings
    'related_paper': "Related Paper(s)",
    'evaluation': 'Evaluation Results',
    # don't include titles
    'extra_analysis': None,
}

# types of functions; I get the strings mixed ups easily
CLASSIFIER = 'classifier'
GENERATOR = 'generator'
RETRIEVER = 'retriever'
RANKER = 'ranker'

# different modes
M_gen = 'gen'
M_edit = 'editing'
M_final = 'final'

# dictionary of all models with their path as the key
all_model_dicts = {model['path']: model for model in model_list}
# dictionary of all tasks with their task field as the key
all_task_dicts = {task['task']: task for task in task_list}


# default messages for certain sections
default = {
    "intended_use": "This model is intended for the use of....",
    "privacy": "This model has the following privacy concerns....",
}

# set containing internal tasks where we can just remove "internal:" and it will be okay
internal_remove_tasks = {
    "internal:blended_skill_talk",
    "internal:light_dialog",
    "internal:light_dialog",
    "internal:eli5",
    "internal:igc",
    "internal:safety:wikiToxicComments",
    "internal:safety:adversarial",
    "internal:safety:boring",
}

# dictionary mapping original task to new location, if they exist
remap_task = {
    "internal:convai2_review": None,
    "internal:safety:boringConvAI2Review": None,
    "internal:comment_battle:ImageDialogGenerationTeacher": None,
    "internal:safety:adversarialConvAI2Review": None,
    "internal:new_reddit:small": None,
}

data_stats_folder = 'data_stats'

#######################################
# Printing/String related functions
#######################################


def create_dropdown(title, dropdown_list):
    newline = '\n'
    return f"<details> {newline} <summary> {title} </summary>{newline} <br>{newline}{newline}{newline.join(dropdown_list)}{newline}</details>"


def taskname(task, task_dict):
    if task_dict:
        return task_dict.get('task')
    task = task.replace('_', ' ')
    task_splitted = task.split(':')
    if len(task_splitted) == 2:
        return task_splitted[0] + ' (' + task_splitted[1] + ')'
    return task.replace(':', ': ')


def format(info):
    if type(info) == dict:
        return json.dumps(info, sort_keys=True, indent=4)
    return str(info)


def process_task(task, ignore_files=True):
    # processing tasks so that no arguments are included
    # unless it's a fromfile or jsonfile one
    splitted = task.split(':')
    stop = len(splitted)
    if 'fromfile:' not in task and 'jsonfile:' not in task:
        for i in range(len(splitted)):
            if '=' in splitted[i]:
                stop = i
                break
    actual_task = ':'.join(splitted[:stop])

    # using actual task, figure out if it should be redirected
    if actual_task in internal_remove_tasks:
        return task.replace('internal:', '')
    if actual_task in remap_task:
        return remap_task[actual_task] + ':'.join(splitted[stop:])
    if 'fromfile:' in task or 'jsonfile:' in task or 'internal:' in task:
        return None if ignore_files else task
    return task


def possible_and_statement(lis):
    L = len(lis)
    if L == 0:
        return ''
    if L == 1:
        return lis[0]
    if L == 2:
        return lis[0] + ' and ' + lis[1]
    return ', '.join(lis[:-1]) + ', and ' + lis[-1]


def create_warning(missing, line_width=150):
    return "> :warning: " + missing + ":warning:"


def extra_msg(string, color='green', sep='-', sep2='*', line_width=80):
    msg = sep * line_width + '\n'
    msg += colorize(f' {string} '.center(line_width, sep2), color)
    msg += '\n' + sep * line_width + '\n'
    return msg


def extra_special_print(string, color='green', sep='-', sep2='*', width=80):
    print(extra_msg(string, color, sep, sep2, width))


def metric_format(metric):
    return '`' + re.sub(r'_+', ' ', metric) + '`'


def to_zoo(opt, model_path):
    # changes absolute model path to sth zoo:model if possible
    zoo = os.path.join(opt['datapath'], 'models') + '/'
    return model_path.replace(zoo, 'zoo:')


def get_dstats_fname(folder_to_save, task):
    taskname = task
    if '/' in task:
        taskname = task.split('/')[-1].split('.')[0]
    fname = '_'.join([taskname, 'train']) + '.json'
    return os.path.join(folder_to_save, data_stats_folder, fname)


def get_saving_err_msg(task, operation, command, e):
    return f'Unfortunately, {operation} for {task} was not successful.\nPlease try running it yourself and debugging.\nEquivalent command tried is\n{command}\n\nError Message:\n{e}\n'


# def process_metrics(report):
#     '''
#     processes a dictionary of metrics for quantitative analysis
#     so that only the type and gender-neutral is left and r-eplaces general
#     with all
#     '''
#     new_report = {}
#     for metric in report:
#         splitted = metric.split('/')
#         if len(splitted) == 1:
#             new_report[]


#################################
# Table-Related Functions
#################################
def make_data_row(task, tname, stats, metrics, suffixes):
    row = [tname]
    for metric in metrics:
        curr_suf = suffixes
        if metric == 'utterances':
            curr_suf = [suffixes[0]]
        for suffix in curr_suf:
            key = suffix + '/' + metric.replace(' ', '_')
            item = stats.get(key, 'n/a')
            if isinstance(item, float):
                item = '{:.3f}'.format(item)
            row.append(str(item))
    row.append(f'`parlai dd -t {task} --dt train`')
    return row


def make_table_header(table_header, align=None, extra='|'):
    align_bars = '---'
    if align == 'center':
        align_bars = ':---:'
    elif align == 'right':
        align_bars = '---:'
    elif align == 'left':
        align_bars = ':---'
    header = extra + ' | '.join(table_header)
    line = ' | '.join([align_bars] * len(table_header))
    return [header, line]


def make_md_table(rows, cols, align='center', extra='|'):
    """
    expect the columns to be a list of headers rows to be a list of lists.
    """
    table = make_table_header(cols, align, extra)
    for row in rows:
        table.append(' | '.join(row))
    return table


def make_html_table(rows, header):
    table = ['<table><tr><th>' + '</th><th>'.join(header) + '</th></tr>']
    for row in rows:
        table.append('<tr><td>' + '</td><td>'.join(row) + '</td></tr>')
    table.append('</table>')
    return "\n".join(table)


# def make_exs_tables(f_contents):
#     """
#     creates a table within a table for the number of examples in each subgroup
#     `f_contents` should be a dictionary that contains the quantative analysis results of
#     this format: {(datatype, subgroup): file_content}

#     Sample table could look like this:
#     |gender | gender2|
#     :--:|:--:
#     |<table>
#     <tr><th> Datatype </th><th> all </th><th>female</th><th>gender-neutral</th><th>male</th></tr>
#     <tr><td>test</td><td>6000</td><td>1459</td><td>2907</td><td>1634</tr></table>
#     |<table>
#     <tr><th> Datatype </th><th> all </th><th>female</th><th>gender-neutral</th><th>male </th></tr>
#     <tr><td>test</td><td>6000</td><td>124</td><td>5277</td><td>599</tr></table>|
#     """
#     # subgroups keys are the title of the first level of table
#     header = list({subgroup_key for _, subgroup_key in f_contents})

#     subgroups = {subgroup: set() for subgroup in header}

#     # creating table
#     tables = []
#     subgroups_keys = {subgroup: [] for subgroup in header}
#     for dt, subgroup in f_contents:
#         subgroups_keys[subgroup].append(dt)
#     for subgroup_key in header:
#         #  first get subgroups as level 2 table headers
#         exs_metric_keys = set()
#         for i, dt in enumerate(subgroups_keys[subgroup_key]):
#             report = f_contents[(dt, subgroup_key)]['report']
#             exs_metric_keys = {metric for metric in report if 'exs' in metric}
#             exs_metric_keys = sorted(list(exs_metric_keys))
#             # headers
#             if i == 0:
#                 header = []
#                 for key in exs_metric_keys:
#                     header.append('all')
#                     if key != 'exs':
#                         key = key.replace('.json', '')
#                         header[-1] = key.split('/')[-2].split('_')[-1]
#                 table += "<tr><th>Datatype</th><th>"
#                 table += f"{'</th><th>'.join(header)} </th></tr>"
#             # actual row
#             row_content = [str(report[mkey]) for mkey in exs_metric_keys]
#             table += f"<tr><td>{dt}</td><td>{'</td><td>'.join(row_content)}</tr>"
#         table += '</table>'
#     return table1_header + table + '|'

#################################
# Setup-related functions
#################################


def setup_args(parser=None) -> ParlaiParser:
    if parser is None:
        parser = ParlaiParser(True, True, 'Automatically make the model card')
        parser.add_arg(
            '--model-type',
            '-mt',
            type=str,
            default=None,
            choices=['ranker', 'generator', 'classifier', 'retriever'],
        )
        parser.add_arg(
            '--folder-to-save',
            '-fts',
            '-ftsaved',
            type=str,
            default="model_card_folder",
            help='folder to save the model card and related contents (ie. graphs)',
        )
        parser.add_arg(
            '--mode',
            type=str,
            default='editing',
            help='possible modes: gen (generation), editing, final.\nIn addition, for gen mode, we can also add the following to specify which exact reports to run: data_stats, eval, safety, sample, and quant)\n For instance, --mode gen:data_stats:eval',
        )
        # FIXME
        # parser.add_arg(
        #     '--user-section-list',
        #     '-exjson',
        #     '-exj',
        #     type=str,
        #     default=None,
        #     help='The json file which contains the user section lists; see documentation for details on formatting',
        # )
        parser.add_arg(
            '--evaltask',
            '-et',
            type=str,
            default=None,
            help='evaluation tasks to use in place of the one specified by model.opt; will only affect `gen` mode',
        )
        parser.add_arg(
            '--evaluation-report-file',
            '-eval-rf',
            type=str,
            default=None,
            help="evaluation report file",
        )
        parser.add_arg(
            '--extra-args-path',
            '-exargs',
            type=str,
            default=None,
            help='path to .json file with extra arguments used for different stages of report generation and later for quantitative analyses section generation; please do NOT use the shortened format (ie. t=<task>); check documentation for more info',
        )
        parser.add_arg(
            '--quantitative-report-files',
            '-quant-rfs',
            type=str,
            default=[],
            nargs='*',
            help='quantitative report file (with different subgroups); if multiple, please separate with comma, and (optional) also add a field in the report file stating what kind of subgroup it is; note that this is only applicable for classifier type models',
        )
        parser.add_arg(
            '--paper',
            type=str,
            default=':',
            help='the key to split the paper name and link on; for instance if the key is `:`, then the input for --papers would be something like name:paper,name:paper',
        )
        parser.add_arg(
            '--include-misc',
            type='bool',
            default=True,
            help='whether to include the miscellaneous dropdown (fields that were not included in other dropdowns); by default, the value is True.',
        )
        return parser


def decide_model_type(opt, model_dict):
    # decide from user input
    if opt.get('model_type'):
        return opt['model_type']

    # fields to check and key words that match to a model type
    check_fields = ('agent', 'title', 'path', 'description')
    key_words = {
        'ranker': RANKER,
        'classifier': CLASSIFIER,
        'generator': GENERATOR,
        'retrieval': RETRIEVER,
        'retriever': RETRIEVER,
    }

    # decide from model_dict
    for key, model_type in key_words.items():
        for field in check_fields:
            if model_dict.get(field) and key in model_dict.get(field):
                return model_type


#################################
# Report Saving Functions
#################################


def save_data_stats(opt, tasks, args):
    folder = os.path.join(opt['folder_to_save'], data_stats_folder)
    err_mgs = []
    os.makedirs(folder, exist_ok=True)
    for task in tasks:
        fname = get_dstats_fname(opt['folder_to_save'], task)
        command_info = f"running `data_stats.DataStats.main(task=task, datatype='train', **args)`\n(args either empty dictionary or what's passed in via `--extra-args-path` for `data_stats_args`) and saving its output in {fname}. \n[Note that running it in commandline doesn't allow saving]"
        extra_special_print(command_info)
        try:
            task_stats = data_stats.DataStats.main(task=task, datatype='train', **args)
            with open(fname, 'w+') as f:
                json.dump(task_stats, f, default=lambda x: x.value())
        except Exception as e:
            msg = get_saving_err_msg(task, 'saving data stats', command_info, e)
            extra_special_print(msg, color='red')
            err_mgs.append(msg)
    return err_mgs if len(err_mgs) > 0 else None


def save_eval(opt, eval_tasks, args):
    metric_fname = os.path.join(opt['folder_to_save'], 'eval_results.json')
    base_args = [f'--{k} {v}' for k, v in args.items()]
    command = f"parlai em -mf {opt['model_file']} -t {','.join(eval_tasks)} -dt test --aggregate-micro True -rf {metric_fname} {' '.join(base_args)}"
    extra_special_print(command)

    try:
        _ = eval_model.EvalModel.main(
            model_file=opt['model_file'],
            task=','.join(eval_tasks),
            datatype='test',
            aggregate_micro=True,
            report_filename=metric_fname,
            **args,
        )
    except Exception as e:
        msg = get_saving_err_msg(','.join(eval_tasks), 'the evaluation', command, e)
        extra_special_print(msg, color='red')
        return [msg]


def save_safety_bench(opt, args):
    # setup wrapper name if it doesn't exist in args
    wrapper = args.get('wrapper')
    if wrapper is None:
        wrapper = opt['model_file'].split('/')[-2]
        # ie. changes blender_90M to blenderbot_90M
        if 'blender_' in wrapper:
            size = wrapper.split('_')[-1]
            wrapper = 'blenderbot_' + size
        args['wrapper'] = wrapper

    folder_name = os.path.join(opt['folder_to_save'], 'safety_bench_res')
    os.makedirs(folder_name, exist_ok=True)
    base_args = [f"--{key} {val}" for key, val in args.items()]
    command = f"python projects/safety_bench/run_unit_tests.py --log-folder {folder_name} {' '.join(base_args)}"
    extra_special_print(command)

    try:
        SafetyUnitTests.main(log_folder=folder_name, **args)
    except Exception as e:
        msg = get_saving_err_msg(wrapper, 'generating safety bench results', command, e)
        msg += '\n\nPlease checkout https://github.com/facebookresearch/ParlAI/tree/master/projects/safety_bench for exact details about implementation of wrapper.'
        extra_special_print(msg, color='red')
        return [msg]


def save_sample(gen_opt, train_task, model_opt, args):
    opt = copy.deepcopy(model_opt)
    opt['num_examples'] = 1
    opt['batchsize'] = 1
    opt['use_test_set'] = False
    opt['fixed_response'] = None
    opt['balance_data'] = model_opt.get('balance_data')
    opt['task'] = train_task
    opt.update(args)
    # for consistent results
    if 'ordered' not in opt['datatype'] and 'train' in opt['datatype']:
        opt['datatype'] = f"{opt['datatype']}:ordered"

    if gen_opt['verbose']:
        extra_special_print('opt used for generating sample input/output')
        opt.log()
        print('using this task for sample input', opt['task'])

    # creating agents and worlds
    agent = FixedResponseAgent(opt)
    world = create_task(opt, agent)

    # getting text and saving labels
    world.parley()
    act = world.acts[0]
    fname = os.path.join(gen_opt['folder_to_save'], 'sample.json')
    with open(fname, 'w+') as f:
        json.dump(act, f)


def save_reports(opt, jobs, args):
    """
    jobs should be {job names: (arguments)} args should be the {job names + '_args' or
    '_qargs': arguments read from user}

    Possible job names: data_stats, eval, safety_bench, sample, quant
    """
    err_msgs = []
    general_args = args['general']
    for job_name, other_args in jobs.items():
        func = 'save_' + job_name
        arg_key = job_name + '_args'
        if args.get(arg_key) is None:
            arg_key = job_name + '_qargs'
        user_args = copy.deepcopy(general_args)
        user_args.update(args.get(arg_key, {}))
        errs = eval(func)(opt, *(other_args[1:]), user_args)
        if errs:
            err_msgs.extend(errs)

    if len(err_msgs) > 0:
        extra_special_print('There are some errors to be resolve', color='blue')
        for msg in err_msgs:
            extra_special_print(msg, color='red', sep=' ')
    else:
        extra_special_print('No error messages were met :)')


###################################
# Script for generating model card
###################################


@register_script('generate_model_card', aliases=['gmc'])
class GenerateModelCard(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        self.opt.log()
        self.verbose = self.opt['verbose']
        self.mode = self.opt['mode'].split(':')[0]

        if self.mode not in {M_gen, M_edit, M_final}:
            raise RuntimeError('The mode' + self.mode + 'was unrecognized.')

        self.general_setup()
        if self.mode == M_gen:
            self._set_evaltask()
            jobs, args = self._gen_jobs()
            save_reports(self.opt, jobs, args)
        else:
            # card setting up
            self._set_sections_info()
            self._get_eval_res()
            self._set_validation_metric()

            # creating & saving content
            self.create_model_card()
            self.save_model_card()

    ##########################################
    # general setup-related class functions
    ##########################################

    def general_setup(self):
        # setting up for saving in the correct folder
        os.makedirs(self.opt['folder_to_save'], exist_ok=True)
        self._add_user_model_tasks()
        self._set_model_dict()
        self._set_model_opt()
        self._set_train_tasks()
        # actually deciding model type
        self.model_type = decide_model_type(self.opt, self.model_dict)

    def _add_user_model_tasks(self):
        # get relevant args using `get_args`
        keywords = {'extra_models': {}, 'extra_tasks': {}}
        user_input = self.get_args(keywords)
        # adding user input to all_model_dicts and all_task_dicts
        for path, dict in user_input['extra_models'].items():
            all_model_dicts[path].update(dict)
        for task, dict in user_input['extra_models'].items():
            all_task_dicts[task].update(dict)

    def _set_model_dict(self):
        # find dictionaries from model_list.py
        mf, self.model_dict = (self.opt['model_file'], {})
        exp_path = to_zoo(self.opt, mf)
        if self.verbose:
            print('expected path in model list:', exp_path, 'or', mf)
        if all_model_dicts.get(exp_path):
            self.model_dict.update(all_model_dicts[exp_path])
        elif all_model_dicts.get(mf):
            self.model_dict.update(all_model_dicts[mf])

    def _set_model_opt(self):
        # reading model.opt
        fopt = self.opt['model_file'] + '.opt'
        if not os.path.isfile(fopt):
            raise RuntimeError(f"The {fopt} can't be found")
        try:
            with open(fopt, 'rb') as f:
                model_opt = json.load(f)
            self.model_opt = Opt(model_opt)
        except UnicodeDecodeError:
            raise RuntimeError(fopt + " isn't in the expected format")

        # override with the override field
        self.model_opt.update(self.model_opt['override'])

        if self.verbose:
            extra_special_print('model.opt')
            self.model_opt.log()
        # for cases where the model wasn't updated, like transformer_classifier
        if self.opt['model']:
            self.model_opt['model'] = self.opt['model']

    def _set_train_tasks(self):
        # setting train tasks
        train_tasks = self.opt.get('task')
        if not train_tasks:
            train_tasks = self.model_opt.get('task', '')
        while len(train_tasks) == 0:
            msg = "Please enter training dataset/test (none were passed in or found in model.opt):"
            train_tasks = input(msg)

        # process tasks from any internal to external
        self.train_tasks, tmp = ([], train_tasks.split(','))
        if self.verbose:
            extra_special_print(f'tasks before: {tmp}')

        for task in tmp:
            processed = process_task(task)
            if processed:
                self.train_tasks.append(processed)

        if self.verbose:
            extra_special_print(f'tasks after: {self.train_tasks}')

        if self.mode != M_gen:
            # only add the tasks that do have a stats file
            self.train_tasks, train_tasks = ([], self.train_tasks)
            for task in train_tasks:
                fname = get_dstats_fname(self.opt['folder_to_save'], task)
                if os.path.isfile(fname):
                    self.train_tasks.append(task)

    ##########################################
    # generation setup-related class functions
    ##########################################

    def _set_evaltask(self):
        # setting eval tasks
        self.eval_tasks = self.train_tasks
        if self.opt.get('evaltask'):
            self.eval_tasks = self.opt['evaltask'].split(',')
        elif self.model_opt.get('evaltask'):
            self.eval_tasks = self.model_opt['evaltask'].split(',')

    def _gen_jobs(self):
        splitted = self.opt['mode'].split(':')
        # job name: (default struct for getting arguments, any other arguments to pass to function)
        all_jobs = {
            'data_stats': ({}, self.train_tasks),
            'eval': ({}, self.eval_tasks),
            'safety_bench': [{}],
            'sample': ({}, self.train_tasks[0], self.model_opt),
        }
        if len(splitted) > 1:
            jobs = {job: all_jobs[job] for job in splitted[1:] if all_jobs.get(job)}
        else:
            jobs = copy.deepcopy(all_jobs)
        if self.model_type != GENERATOR:
            del all_jobs['safety_bench']
        key_defaults = {(job + '_args'): params[0] for job, params in jobs.items()}
        # adding a general field for later use
        key_defaults['general'] = {}
        args = self.get_args(key_defaults)
        return jobs, args

    ##########################################
    # model card setup-related class functions
    ##########################################

    def _set_sections_info(self):
        key_defaults = {'user_section_list': [], 'dropdowns': {}}
        args = self.get_args(key_defaults)

        # set up dropdowns
        self.hyperparams = default_hyperparams
        if len(args['dropdowns']) > 0:
            self.hyperparams = args['dropdowns']

        # set up section_list
        self.section_list = section_list
        if args.get('user_section_list'):
            self.section_list = args['user_section_list']
            for i, section in enumerate(self.section_list):
                if section not in defined_sections:
                    self.section_list[i] = USER_SYM_SECTION + section

    def _set_validation_metric(self):
        self.valid_metric = self.model_opt.get('validation_metric', '')
        metrics = {metric.split('/')[-1] for metric in self.eval_results.keys()}
        color_msg = "\nValidation metric isn't given (correctly)..here're the possible metrics:\n"
        msg = colorize(color_msg, 'red') + '-' * 100 + '\n' + "\n".join(metrics) + '\n'
        msg += colorize('Please enter a validation metric: ', 'red')
        while self.valid_metric not in metrics:
            self.valid_metric = input(msg).strip()

    def _get_eval_res(self):
        if self.opt.get('evaluation_report_file'):
            fname = self.opt['evaluation_report_file']
        else:
            # read in report made by `gen` mode
            fname = os.path.join(self.opt['folder_to_save'], 'eval_results.json')
        try:
            with open(fname) as json_file:
                tmp = json.load(json_file)
        except Exception:
            raise RuntimeError(
                f'The {fname} does not exist; please run in `--mode gen` to generate this file for evaluation section or generate your own metric results.'
            )
        self.eval_tasks = tmp['opt']['task'].split(',')
        self.eval_results = tmp['report']

    ##########################################
    # main creating & saving model card funcs
    ##########################################
    def create_model_card(self):
        self.section_contents = {}

        for section in self.section_list:
            extra_special_print(f'creating section {section}')
            # remove subsection info before subsection
            func_name = section.split(':_')[-1]
            # user written sections
            if USER_SYM_SECTION in section:
                if self.verbose:
                    print('User added/should add this section:', section)
                section_title = func_name.replace(USER_SYM_SECTION, '')
                self.section_contents[section] = self.user_defined_sec(section_title)
            # regular sections
            else:
                self.section_contents[section] = eval("self." + func_name)()

    def _create_content(self):
        contents = ['# ' + self.model_dict.get('title', 'Missing Model Title')]
        for section in self.section_list:
            # getting section contents + creating header if necessary
            section_title = section.split(':_')[-1]

            if self.section_contents.get(section):
                header_ct = section.count(':_') + 2
                section_title = section_title.replace(USER_SYM_SECTION, '')

                if section in special_section:
                    section_title = special_section[section]
                else:
                    section_title = section_title.replace('_', ' ').title()

                header = ''
                if section != 'model_details' and section_title:
                    header = '#' * header_ct + ' ' + section_title
                contents.append(header + '\n\n' + self.section_contents[section])
                extra_special_print(f'finished appending content for {section}')
        BACK_TO_TOP = f"\n[back-to-top]({contents[0].replace(' ', '-')})\n"
        # add a back to top at the very end
        contents.append(BACK_TO_TOP)
        return contents

    def save_model_card(self, model_card_name='model_card.md'):
        # setting up for saving
        fname = os.path.join(self.opt['folder_to_save'], model_card_name)

        # writing model card
        with open(fname, 'w+') as f:
            f.write('\n\n'.join(self._create_content()))
        extra_special_print(fname + ' was saved.', sep2=' ', color='blue')

    ##########################################
    # section functions
    ##########################################

    def model_details(self):
        descrip = self._find_by_key('description')
        contents = [descrip] if descrip else [create_warning('missing description')]
        li_list = list(map(self._search_make_li, model_details_struct))
        contents += list(filter(None, li_list))
        return '\n'.join(contents)

    def quick_usage(self):
        if self.model_dict.get('example'):
            command, extra = (self.model_dict['example'], '')
        elif self.model_opt.get('path') and self.model_opt.get('task'):
            # auto-generate a command
            command = f"parlai interactive -mf {self.model_opt['path']} -t {self.model_opt['task']}"
            msg = "This command was auto-generated by the script; please go to [GitHub Issues page](https://github.com/facebookresearch/ParlAI/issues) if there are issues."
            extra = create_warning(msg)
        else:
            msg = "Could not find an example in model_list.py for this model and could not find the `task` and `path` in model.opt, so cannot add quick usage"
            print(colorize(msg, 'red'))
            if self.mode == M_edit:
                return create_warning('missing quick usage/sample command')
            else:
                return None
        return extra + '```\n' + command + '\n```'

    def sample_input_and_output(self):
        fname = os.path.join(self.opt['folder_to_save'], 'sample.json')
        with open(fname, 'rb') as f:
            sample = json.load(f)
        ignore_fields = {'id', 'episode_done'}
        contents = []
        for k, v in sample.items():
            v = v.strip() if isinstance(v, str) else None
            if k not in ignore_fields and v:
                contents.append(f"[{k}]: {v}")
        return '```\n' + '\n\n'.join(contents) + '\n```'

    def datasets_used(self):
        pass

    def evaluation(self):
        pass

    def extra_analysis(self):
        pass

    def feedback(self):
        return "We would love any feedback about the model (or the model card script)! Feel free to report any issues or unexpected findings using our [GitHub Issues page](https://github.com/facebookresearch/ParlAI/issues):)"

    def hyperparameters(self):
        if self.verbose:
            extra_special_print("model_opt:", color='yellow')
            self.model_opt.log()

        # try to check for init model and see if there are any differences
        try:
            self.init_model = Opt.load(self.model_opt['init_model'])
            extra_func = self._check_init_model_keys
            maybe_keys = ['dropout']
        except Exception:
            maybe_keys = []
            extra_func = None

        not_dropdowns = ['always_include', 'always_exclude', 'maybe_special']
        dropdowns = [k for k in self.hyperparams if k not in not_dropdowns]
        dropdowns.insert(0, 'top_not_dropdown')
        dropdowns.append('miscellaneous')

        # keys to exclude + remove any values with '/'
        rest_keys = set(
            [k for k, v in self.model_opt.items() if v and '/' not in str(v)]
        )
        rest_keys.difference_update(
            set(self.hyperparams.get('always_exclude', ())[OTHER])
        )
        all_content = []
        for title in dropdowns:
            values = self.hyperparams.get(title)
            if title == 'top_not_dropdown':
                special = self.hyperparams['maybe_special'][OTHER]
                keys = self.hyperparams['always_include'][OTHER] + list(special.keys())
                default_value = 'Not specified'
                process_func = [None] * (len(keys)) + [extra_func] * len(maybe_keys)
                process_func += [special[k] for k in keys[-len(special) :]]
            elif title != 'miscellaneous':
                common, keys = values
                if common is None or common == 'None':
                    commmon_keys = []
                elif callable(common):
                    commmon_keys = [key for key in rest_keys if common(key)]
                else:
                    commmon_keys = [key for key in rest_keys if common in key]
                keys.extend(commmon_keys)
                default_value = None
                process_func = [None] * len(keys)
            elif self.opt['include_misc']:
                keys = rest_keys
                default_value = None
                process_func = [None] * len(keys)
            else:
                break  # breaking b/c we don't need to add 'misc' & 'misc' should be the last item

            L = len(keys)
            default_values = [default_value] * L
            before_value = [f'`{key}`: `' for key in keys]
            after_value = ['`'] * L
            struct = zip(keys, default_values, before_value, after_value, process_func)
            content = [self._search_make_li(item, warning=False) for item in struct]
            content = list(filter(None, content))
            if len(content) > 0 and title != 'top_not_dropdown':
                all_content.append(create_dropdown(title, content))
            else:
                all_content.insert(0, '\n'.join(content))
            rest_keys.difference_update(keys)

        return '\n'.join(all_content)

    def user_defined_sec(self, section_title):
        pass

    ##########################################
    # misc/other helper functions
    ##########################################

    def _check_init_model_keys(self, r, k):
        return r if self.init_model[k] != self.model_opt[k] else None

    def get_args(self, key_defaults):
        """
        Finds the arguments from extra_args_path based on the list of keywords, and
        replaces or overrides defaults.

        Possible keywords for key_defaults: data_stats_args / eval_args / safety_args /
        label_qargs / model_qargs / eval_qargs / section_qargs / user_sections /
        extra_tasks / extra_models
        """
        args = copy.deepcopy(key_defaults)
        user_args = self.opt['extra_args_path']
        if user_args is None:
            user_args = os.path.join(self.opt['folder_to_save'], 'args.json')
        try:
            with open(user_args, 'rb') as f:
                all_args = json.load(f)
            for key in key_defaults:
                if all_args.get(key) and isinstance(args[key], dict):
                    args[key].update(all_args.get(key, {}))
                elif all_args.get(key):
                    args[key] = all_args.get(key)
        except Exception as e:
            extra_special_print(e, 'red')
        return args

    def _find_by_key(self, key):
        value = self.model_dict.get(key)
        if value is None and self.model_opt.get(key):
            value = self.model_opt[key]
        return value

    def _search_make_li(self, li_tuple, pass_key=False, warning=True):
        """
        makes a markdown list item when we need to search in either model_dict or
        model_opt for a value;

        expects this input format for li_tuple:
        (key to look for in model_dict or model_opt,
         default value/function,
         string before value,
         string after value,
         optional: processing func on key if found
         optional: number of tabs; default is 0,

        pass_key specifies whether or not to pass the key to
        the lambda function for postprocessing

        if warning is False, the warning will be supressed even
        if the mode is editing
        """
        # make sure to have at least four values in the tuple
        if len(li_tuple) < 4:
            return None

        # get tuple into names
        key, default, before, after = li_tuple[:4]
        func, tab_num = (None, 0)
        if isinstance(li_tuple[-1], int):
            tab_num = li_tuple[-1]

        if len(li_tuple) > 4 and callable(li_tuple[4]):
            func = li_tuple[4]

        if self.verbose:
            print(colorize(f'searching for {key}', 'yellow'))

        # looking for the value using key
        value = self._find_by_key(key)

        # default case
        if value is None:
            value = default() if callable(default) else default
            if type(value) is not str:
                if self.mode == M_edit and warning:
                    warning = f'{key} must be a string...currently it looks like this: {format(value)}'
                    msg = create_warning(warning)
                    extra_special_print(msg, 'red')
                    return msg
                return None

        # postprocessing the key
        if func and pass_key:
            value = func(value, key)
        elif func:
            value = func(value)
        if value is not None:
            # checking if the last one is a number for the tab number
            value = '\t' * tab_num + ' '.join(('-', before, format(value), after))
        return value


if __name__ == '__main__':
    GenerateModelCard.main()
