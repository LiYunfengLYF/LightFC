import multiprocessing
import os
import sys
import time
from collections import OrderedDict
from datetime import timedelta
from itertools import product

import numpy as np
import torch

from lib.test.evaluation import Sequence, Tracker
from lib.utils.list_tools import split_list


def _save_tracker_output(seq: Sequence, tracker: Tracker, output: dict):
    """Saves the output of the tracker."""

    if not os.path.exists(tracker.results_dir):
        print("create tracking result dir:", tracker.results_dir)
        os.makedirs(tracker.results_dir)

    if not os.path.exists(os.path.join(tracker.results_dir, seq.dataset)):
        os.makedirs(os.path.join(tracker.results_dir, seq.dataset))

    '''2021.1.5 create new folder for these two datasets'''
    base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)

    def save_bb(file, data):
        tracked_bb = np.array(data).astype(int)
        np.savetxt(file, tracked_bb, delimiter='\t', fmt='%d')

    def save_time(file, data):
        exec_times = np.array(data).astype(float)
        np.savetxt(file, exec_times, delimiter='\t', fmt='%f')

    def save_score(file, data):
        scores = np.array(data).astype(float)
        np.savetxt(file, scores, delimiter='\t', fmt='%.2f')

    def _convert_dict(input_dict):
        data_dict = {}
        for elem in input_dict:
            for k, v in elem.items():
                if k in data_dict.keys():
                    data_dict[k].append(v)
                else:
                    data_dict[k] = [v, ]
        return data_dict

    for key, data in output.items():
        # If data is empty
        if not data:
            continue

        if key == 'target_bbox':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_boxes':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_boxes.txt'.format(base_results_path, obj_id)
                    save_bb(bbox_file, d)
            else:
                # Single-object mode
                bbox_file = '{}_all_boxes.txt'.format(base_results_path)
                save_bb(bbox_file, data)

        if key == 'all_scores':
            if isinstance(data[0], (dict, OrderedDict)):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    bbox_file = '{}_{}_all_scores.txt'.format(base_results_path, obj_id)
                    save_score(bbox_file, d)
            else:
                # Single-object mode
                print("saving scores...")
                bbox_file = '{}_all_scores.txt'.format(base_results_path)
                save_score(bbox_file, data)

        elif key == 'time':
            if isinstance(data[0], dict):
                data_dict = _convert_dict(data)

                for obj_id, d in data_dict.items():
                    timings_file = '{}_{}_time.txt'.format(base_results_path, obj_id)
                    save_time(timings_file, d)
            else:
                timings_file = '{}_time.txt'.format(base_results_path)
                save_time(timings_file, data)


def _show_finished_seq_num(seq: Sequence, tracker: Tracker, len_dataset):
    base_results_path = os.path.join(tracker.results_dir, seq.dataset)
    finished_num = int(len(os.listdir(base_results_path)) / 2)
    len_dataset = int(len_dataset)
    return f'{finished_num} / {len_dataset}'


def run_sequence(seq: Sequence, tracker: Tracker, debug=False, num_gpu=8, env_num=0, len_dataset=0):
    """Runs a tracker on a sequence."""
    '''2021.1.2 Add multiple gpu support'''
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(gpu_id)
    except:
        pass

    def _results_exist():
        if seq.object_ids is None:
            base_results_path = os.path.join(tracker.results_dir, seq.dataset, seq.name)
            bbox_file = '{}.txt'.format(base_results_path)

            return os.path.isfile(bbox_file)
        else:
            bbox_files = ['{}/{}_{}.txt'.format(tracker.results_dir, seq.name, obj_id) for obj_id in seq.object_ids]
            missing = [not os.path.isfile(f) for f in bbox_files]
            return sum(missing) == 0

    if _results_exist() and not debug:
        print(
            f'  [{seq.dataset}]\t[{_show_finished_seq_num(seq, tracker, len_dataset)}]\t[{tracker.name}]\t{tracker.parameter_name}\tFPS:{-1:4d}\tSequence: {seq.name}')
        return

    # print('Tracker: {} {} {} ,  Sequence: {}'.format(tracker.name, tracker.parameter_name, tracker.run_id, seq.name))

    if debug:
        output = tracker.run_sequence(seq, debug=debug, env_num=env_num)
    else:
        try:
            sys.stdout = open('test_output.txt', 'w')
            output = tracker.run_sequence(seq, debug=debug, env_num=env_num)
            sys.stdout = sys.__stdout__
        except Exception as e:
            print(e)
            return

    sys.stdout.flush()

    if isinstance(output['time'][0], (dict, OrderedDict)):
        exec_time = sum([sum(times.values()) for times in output['time']])
        num_frames = len(output['time'])
    else:
        exec_time = sum(output['time'])
        num_frames = len(output['time'])

    if not debug:
        _save_tracker_output(seq, tracker, output)

        print(
            f'  [{seq.dataset}]\t[{_show_finished_seq_num(seq, tracker, len_dataset)}]\t[{tracker.name}]\t{tracker.parameter_name}\tFPS:{int(num_frames / exec_time) + 1:4d}\tSequence: {seq.name}')


def run_dataset(dataset, trackers, debug=False, threads=0, num_gpus=8, env_num=0, deploy=True):
    """Runs a list of trackers on a dataset.
    args:
        dataset: List of Sequence instances, forming a dataset.
        trackers: List of Tracker instances.
        debug: Debug level.
        threads: Number of threads to use (default 0).
    """

    print(f'Evaluating {trackers[0].name} {trackers[0].parameter_name} on {len(dataset):5d} sequences')
    dataset_start_time = time.time()

    multiprocessing.set_start_method('spawn', force=True)

    if threads == 0:
        mode = 'sequential'
    else:
        mode = 'parallel'

    if mode == 'sequential':
        for seq in dataset:
            for tracker_info in trackers:
                run_sequence(seq, tracker_info, debug=debug, env_num=env_num, len_dataset=len(dataset))
                # raise

    elif mode == 'parallel':
        param_list = [(seq, tracker_info, debug, num_gpus, env_num, len(dataset)) for seq, tracker_info in
                      product(dataset, trackers)]
        with multiprocessing.Pool(processes=threads) as pool:
            pool.starmap(run_sequence, param_list)
    print('Done, total time: {}'.format(str(timedelta(seconds=(time.time() - dataset_start_time)))), '\n')
