# __ coding: utf-8 __
# author: Li Yunfeng
# data: 2022/12/8 19:14

import numpy as np

from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class UTBDataset(BaseDataset):
    '''UTB180 dataset
    Publication:
    Download the dataset from
    '''

    def __init__(self, env_num):
        super().__init__(env_num)
        self.base_path = self.env_settings.utb_path
        self.sequence_info_list = self._get_sequence_info_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_info_list])

    def _construct_sequence(self, sequence_info):
        sequence_path = sequence_info['path']
        nz = sequence_info['nz']
        ext = sequence_info['ext']
        start_frame = sequence_info['startFrame']
        end_frame = sequence_info['endFrame']

        init_omit = 0
        if 'initOmit' in sequence_info:
            init_omit = sequence_info['initOmit']

        frames = ['{base_path}/{sequence_path}/{frame:0{nz}}.{ext}'.format(base_path=self.base_path,
                                                                           sequence_path=sequence_path, frame=frame_num,
                                                                           nz=nz, ext=ext) for frame_num in
                  range(start_frame + init_omit, end_frame + 1)]

        anno_path = '{}/{}'.format(self.base_path, sequence_info['anno_path'])

        # NOTE: OTB has some weird annos which panda cannot handle
        ground_truth_rect = load_text(str(anno_path), delimiter=('\t'), dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'utb', ground_truth_rect[init_omit:, :], )

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "Video_0001", "path": "Video_0001/imgs", "startFrame": 1, "endFrame": 165, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0001/groundtruth_rect.txt"},

            {"name": "Video_01", "path": "Video_01/imgs", "startFrame": 1, "endFrame": 408, "nz": 4, "ext": "jpg",
             "anno_path": "Video_01/groundtruth_rect.txt", },

            {"name": "Video_0002", "path": "Video_0002/imgs", "startFrame": 1, "endFrame": 484, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0002/groundtruth_rect.txt", },

            {"name": "Video_02", "path": "Video_02/imgs", "startFrame": 1, "endFrame": 56, "nz": 3, "ext": "jpg",
             "anno_path": "Video_02/groundtruth_rect.txt", },

            {"name": "Video_0003", "path": "Video_0003/imgs", "startFrame": 1, "endFrame": 450, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0003/groundtruth_rect.txt", },

            {"name": "Video_03", "path": "Video_03/imgs", "startFrame": 1, "endFrame": 110, "nz": 4, "ext": "jpg",
             "anno_path": "Video_03/groundtruth_rect.txt", },

            {"name": "Video_0004", "path": "Video_0004/imgs", "startFrame": 1, "endFrame": 840, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0004/groundtruth_rect.txt", },

            {"name": "Video_04", "path": "Video_04/imgs", "startFrame": 1, "endFrame": 188, "nz": 4, "ext": "jpg",
             "anno_path": "Video_04/groundtruth_rect.txt", },

            {"name": "Video_0005", "path": "Video_0005/imgs", "startFrame": 1, "endFrame": 451, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0005/groundtruth_rect.txt", },

            {"name": "Video_05", "path": "Video_05/imgs", "startFrame": 1, "endFrame": 447, "nz": 4, "ext": "jpg",
             "anno_path": "Video_05/groundtruth_rect.txt", },

            {"name": "Video_0006", "path": "Video_0006/imgs", "startFrame": 1, "endFrame": 330, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0006/groundtruth_rect.txt", },

            {"name": "Video_06", "path": "Video_06/imgs", "startFrame": 1, "endFrame": 210, "nz": 4, "ext": "jpg",
             "anno_path": "Video_06/groundtruth_rect.txt", },

            {"name": "Video_0007", "path": "Video_0007/imgs", "startFrame": 1, "endFrame": 377, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0007/groundtruth_rect.txt", },

            {"name": "Video_07", "path": "Video_07/imgs", "startFrame": 1, "endFrame": 93, "nz": 3, "ext": "jpg",
             "anno_path": "Video_07/groundtruth_rect.txt", },

            {"name": "Video_0008", "path": "Video_0008/imgs", "startFrame": 1, "endFrame": 377, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0008/groundtruth_rect.txt", },

            {"name": "Video_08", "path": "Video_08/imgs", "startFrame": 1, "endFrame": 210, "nz": 4, "ext": "jpg",
             "anno_path": "Video_08/groundtruth_rect.txt", },

            {"name": "Video_0009", "path": "Video_0009/imgs", "startFrame": 1, "endFrame": 170, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0009/groundtruth_rect.txt", },

            {"name": "Video_09", "path": "Video_09/imgs", "startFrame": 1, "endFrame": 49, "nz": 3, "ext": "jpg",
             "anno_path": "Video_09/groundtruth_rect.txt", },

            {"name": "Video_0010", "path": "Video_0010/imgs", "startFrame": 1, "endFrame": 426, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0010/groundtruth_rect.txt", },

            {"name": "Video_10", "path": "Video_10/imgs", "startFrame": 1, "endFrame": 174, "nz": 4, "ext": "jpg",
             "anno_path": "Video_10/groundtruth_rect.txt", },

            {"name": "Video_0011", "path": "Video_0011/imgs", "startFrame": 1, "endFrame": 294, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0011/groundtruth_rect.txt", },

            {"name": "Video_11", "path": "Video_11/imgs", "startFrame": 1, "endFrame": 58, "nz": 3, "ext": "jpg",
             "anno_path": "Video_11/groundtruth_rect.txt", },

            {"name": "Video_0012", "path": "Video_0012/imgs", "startFrame": 1, "endFrame": 392, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0012/groundtruth_rect.txt", },

            {"name": "Video_12", "path": "Video_12/imgs", "startFrame": 1, "endFrame": 146, "nz": 4, "ext": "jpg",
             "anno_path": "Video_12/groundtruth_rect.txt", },

            {"name": "Video_0013", "path": "Video_0013/imgs", "startFrame": 1, "endFrame": 259, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0013/groundtruth_rect.txt", },

            {"name": "Video_13", "path": "Video_13/imgs", "startFrame": 1, "endFrame": 712, "nz": 4, "ext": "jpg",
             "anno_path": "Video_13/groundtruth_rect.txt", },

            {"name": "Video_0014", "path": "Video_0014/imgs", "startFrame": 1, "endFrame": 341, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0014/groundtruth_rect.txt", },

            {"name": "Video_14", "path": "Video_14/imgs", "startFrame": 1, "endFrame": 239, "nz": 4, "ext": "jpg",
             "anno_path": "Video_14/groundtruth_rect.txt", },

            {"name": "Video_0015", "path": "Video_0015/imgs", "startFrame": 1, "endFrame": 105, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0015/groundtruth_rect.txt", },

            {"name": "Video_15", "path": "Video_15/imgs", "startFrame": 1, "endFrame": 267, "nz": 4, "ext": "jpg",
             "anno_path": "Video_15/groundtruth_rect.txt", },

            {"name": "Video_0016", "path": "Video_0016/imgs", "startFrame": 1, "endFrame": 105, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0016/groundtruth_rect.txt", },

            {"name": "Video_16", "path": "Video_16/imgs", "startFrame": 1, "endFrame": 495, "nz": 4, "ext": "jpg",
             "anno_path": "Video_16/groundtruth_rect.txt", },

            {"name": "Video_0017", "path": "Video_0017/imgs", "startFrame": 1, "endFrame": 105, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0017/groundtruth_rect.txt", },

            {"name": "Video_17", "path": "Video_17/imgs", "startFrame": 1, "endFrame": 601, "nz": 4, "ext": "jpg",
             "anno_path": "Video_17/groundtruth_rect.txt", },

            {"name": "Video_0018", "path": "Video_0018/imgs", "startFrame": 1, "endFrame": 166, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0018/groundtruth_rect.txt", },

            {"name": "Video_18", "path": "Video_18/imgs", "startFrame": 1, "endFrame": 139, "nz": 4, "ext": "jpg",
             "anno_path": "Video_18/groundtruth_rect.txt", },

            {"name": "Video_0019", "path": "Video_0019/imgs", "startFrame": 1, "endFrame": 332, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0019/groundtruth_rect.txt", },

            {"name": "Video_19", "path": "Video_19/imgs", "startFrame": 1, "endFrame": 82, "nz": 3, "ext": "jpg",
             "anno_path": "Video_19/groundtruth_rect.txt", },

            {"name": "Video_0020", "path": "Video_0020/imgs", "startFrame": 1, "endFrame": 552, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0020/groundtruth_rect.txt", },

            {"name": "Video_20", "path": "Video_20/imgs", "startFrame": 1, "endFrame": 123, "nz": 4, "ext": "jpg",
             "anno_path": "Video_20/groundtruth_rect.txt", },

            {"name": "Video_0021", "path": "Video_0021/imgs", "startFrame": 1, "endFrame": 204, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0021/groundtruth_rect.txt", },

            {"name": "Video_0022", "path": "Video_0022/imgs", "startFrame": 1, "endFrame": 270, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0022/groundtruth_rect.txt", },

            {"name": "Video_0023", "path": "Video_0023/imgs", "startFrame": 1, "endFrame": 465, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0023/groundtruth_rect.txt", },

            {"name": "Video_0024", "path": "Video_0024/imgs", "startFrame": 1, "endFrame": 220, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0024/groundtruth_rect.txt", },

            {"name": "Video_0025", "path": "Video_0025/imgs", "startFrame": 1, "endFrame": 363, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0025/groundtruth_rect.txt", },

            {"name": "Video_0026", "path": "Video_0026/imgs", "startFrame": 1, "endFrame": 330, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0026/groundtruth_rect.txt", },

            {"name": "Video_0027", "path": "Video_0027/imgs", "startFrame": 1, "endFrame": 621, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0027/groundtruth_rect.txt", },

            {"name": "Video_0028", "path": "Video_0028/imgs", "startFrame": 1, "endFrame": 585, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0028/groundtruth_rect.txt", },

            {"name": "Video_0029", "path": "Video_0029/imgs", "startFrame": 1, "endFrame": 428, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0029/groundtruth_rect.txt", },

            {"name": "Video_0030", "path": "Video_0030/imgs", "startFrame": 1, "endFrame": 308, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0030/groundtruth_rect.txt", },

            {"name": "Video_0031", "path": "Video_0031/imgs", "startFrame": 1, "endFrame": 235, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0031/groundtruth_rect.txt", },

            {"name": "Video_0032", "path": "Video_0032/imgs", "startFrame": 1, "endFrame": 508, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0032/groundtruth_rect.txt", },

            {"name": "Video_0033", "path": "Video_0033/imgs", "startFrame": 1, "endFrame": 348, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0033/groundtruth_rect.txt", },

            {"name": "Video_0034", "path": "Video_0034/imgs", "startFrame": 1, "endFrame": 176, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0034/groundtruth_rect.txt", },

            {"name": "Video_0035", "path": "Video_0035/imgs", "startFrame": 1, "endFrame": 183, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0035/groundtruth_rect.txt", },

            {"name": "Video_0036", "path": "Video_0036/imgs", "startFrame": 1, "endFrame": 169, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0036/groundtruth_rect.txt", },

            {"name": "Video_0037", "path": "Video_0037/imgs", "startFrame": 1, "endFrame": 227, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0037/groundtruth_rect.txt", },

            {"name": "Video_0038", "path": "Video_0038/imgs", "startFrame": 1, "endFrame": 257, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0038/groundtruth_rect.txt", },

            {"name": "Video_0039", "path": "Video_0039/imgs", "startFrame": 1, "endFrame": 421, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0039/groundtruth_rect.txt", },

            {"name": "Video_0040", "path": "Video_0040/imgs", "startFrame": 1, "endFrame": 199, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0040/groundtruth_rect.txt", },

            {"name": "Video_0041", "path": "Video_0041/imgs", "startFrame": 1, "endFrame": 193, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0041/groundtruth_rect.txt", },

            {"name": "Video_0042", "path": "Video_0042/imgs", "startFrame": 1, "endFrame": 105, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0042/groundtruth_rect.txt", },

            {"name": "Video_0043", "path": "Video_0043/imgs", "startFrame": 1, "endFrame": 107, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0043/groundtruth_rect.txt", },

            {"name": "Video_0044", "path": "Video_0044/imgs", "startFrame": 1, "endFrame": 350, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0044/groundtruth_rect.txt", },

            {"name": "Video_0045", "path": "Video_0045/imgs", "startFrame": 1, "endFrame": 350, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0045/groundtruth_rect.txt", },

            {"name": "Video_0046", "path": "Video_0046/imgs", "startFrame": 1, "endFrame": 350, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0046/groundtruth_rect.txt", },

            {"name": "Video_0047", "path": "Video_0047/imgs", "startFrame": 1, "endFrame": 451, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0047/groundtruth_rect.txt", },

            {"name": "Video_0048", "path": "Video_0048/imgs", "startFrame": 1, "endFrame": 365, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0048/groundtruth_rect.txt", },

            {"name": "Video_0049", "path": "Video_0049/imgs", "startFrame": 1, "endFrame": 296, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0049/groundtruth_rect.txt", },

            {"name": "Video_0050", "path": "Video_0050/imgs", "startFrame": 1, "endFrame": 348, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0050/groundtruth_rect.txt", },

            {"name": "Video_0051", "path": "Video_0051/imgs", "startFrame": 1, "endFrame": 588, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0051/groundtruth_rect.txt", },

            {"name": "Video_0052", "path": "Video_0052/imgs", "startFrame": 1, "endFrame": 1047, "nz": 5, "ext": "jpg",
             "anno_path": "Video_0052/groundtruth_rect.txt", },

            {"name": "Video_0053", "path": "Video_0053/imgs", "startFrame": 1, "endFrame": 643, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0053/groundtruth_rect.txt", },

            {"name": "Video_0054", "path": "Video_0054/imgs", "startFrame": 1, "endFrame": 511, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0054/groundtruth_rect.txt", },

            {"name": "Video_0055", "path": "Video_0055/imgs", "startFrame": 1, "endFrame": 1226, "nz": 5, "ext": "jpg",
             "anno_path": "Video_0055/groundtruth_rect.txt", },

            {"name": "Video_0056", "path": "Video_0056/imgs", "startFrame": 1, "endFrame": 804, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0056/groundtruth_rect.txt", },

            {"name": "Video_0057", "path": "Video_0057/imgs", "startFrame": 1, "endFrame": 419, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0057/groundtruth_rect.txt", },

            {"name": "Video_0058", "path": "Video_0058/imgs", "startFrame": 1, "endFrame": 451, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0058/groundtruth_rect.txt", },

            {"name": "Video_0059", "path": "Video_0059/imgs", "startFrame": 1, "endFrame": 380, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0059/groundtruth_rect.txt", },

            {"name": "Video_0060", "path": "Video_0060/imgs", "startFrame": 1, "endFrame": 476, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0060/groundtruth_rect.txt", },

            {"name": "Video_0061", "path": "Video_0061/imgs", "startFrame": 1, "endFrame": 370, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0061/groundtruth_rect.txt", },

            {"name": "Video_0062", "path": "Video_0062/imgs", "startFrame": 1, "endFrame": 362, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0062/groundtruth_rect.txt", },

            {"name": "Video_0063", "path": "Video_0063/imgs", "startFrame": 1, "endFrame": 324, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0063/groundtruth_rect.txt", },

            {"name": "Video_0064", "path": "Video_0064/imgs", "startFrame": 1, "endFrame": 378, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0064/groundtruth_rect.txt", },

            {"name": "Video_0065", "path": "Video_0065/imgs", "startFrame": 1, "endFrame": 309, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0065/groundtruth_rect.txt", },

            {"name": "Video_0066", "path": "Video_0066/imgs", "startFrame": 1, "endFrame": 405, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0066/groundtruth_rect.txt", },

            {"name": "Video_0067", "path": "Video_0067/imgs", "startFrame": 1, "endFrame": 391, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0067/groundtruth_rect.txt", },

            {"name": "Video_0068", "path": "Video_0068/imgs", "startFrame": 1, "endFrame": 405, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0068/groundtruth_rect.txt", },

            {"name": "Video_0069", "path": "Video_0069/imgs", "startFrame": 1, "endFrame": 417, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0069/groundtruth_rect.txt", },

            {"name": "Video_0070", "path": "Video_0070/imgs", "startFrame": 1, "endFrame": 441, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0070/groundtruth_rect.txt", },

            {"name": "Video_0071", "path": "Video_0071/imgs", "startFrame": 1, "endFrame": 279, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0071/groundtruth_rect.txt", },

            {"name": "Video_0072", "path": "Video_0072/imgs", "startFrame": 1, "endFrame": 318, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0072/groundtruth_rect.txt", },

            {"name": "Video_0073", "path": "Video_0073/imgs", "startFrame": 1, "endFrame": 510, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0073/groundtruth_rect.txt", },

            {"name": "Video_0074", "path": "Video_0074/imgs", "startFrame": 1, "endFrame": 211, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0074/groundtruth_rect.txt", },

            {"name": "Video_0075", "path": "Video_0075/imgs", "startFrame": 1, "endFrame": 275, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0075/groundtruth_rect.txt", },

            {"name": "Video_0076", "path": "Video_0076/imgs", "startFrame": 1, "endFrame": 296, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0076/groundtruth_rect.txt", },

            {"name": "Video_0077", "path": "Video_0077/imgs", "startFrame": 1, "endFrame": 272, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0077/groundtruth_rect.txt", },

            {"name": "Video_0078", "path": "Video_0078/imgs", "startFrame": 1, "endFrame": 151, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0078/groundtruth_rect.txt", },

            {"name": "Video_0079", "path": "Video_0079/imgs", "startFrame": 1, "endFrame": 507, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0079/groundtruth_rect.txt", },

            {"name": "Video_0080", "path": "Video_0080/imgs", "startFrame": 1, "endFrame": 394, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0080/groundtruth_rect.txt", },

            {"name": "Video_0081", "path": "Video_0081/imgs", "startFrame": 1, "endFrame": 183, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0081/groundtruth_rect.txt", },

            {"name": "Video_0082", "path": "Video_0082/imgs", "startFrame": 1, "endFrame": 250, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0082/groundtruth_rect.txt", },

            {"name": "Video_0083", "path": "Video_0083/imgs", "startFrame": 1, "endFrame": 314, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0083/groundtruth_rect.txt", },

            {"name": "Video_0084", "path": "Video_0084/imgs", "startFrame": 1, "endFrame": 224, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0084/groundtruth_rect.txt", },

            {"name": "Video_0085", "path": "Video_0085/imgs", "startFrame": 1, "endFrame": 223, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0085/groundtruth_rect.txt", },

            {"name": "Video_0086", "path": "Video_0086/imgs", "startFrame": 1, "endFrame": 201, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0086/groundtruth_rect.txt", },

            {"name": "Video_0087", "path": "Video_0087/imgs", "startFrame": 1, "endFrame": 178, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0087/groundtruth_rect.txt", },

            {"name": "Video_0088", "path": "Video_0088/imgs", "startFrame": 1, "endFrame": 462, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0088/groundtruth_rect.txt", },

            {"name": "Video_0089", "path": "Video_0089/imgs", "startFrame": 1, "endFrame": 258, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0089/groundtruth_rect.txt", },

            {"name": "Video_0090", "path": "Video_0090/imgs", "startFrame": 1, "endFrame": 150, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0090/groundtruth_rect.txt", },

            {"name": "Video_0091", "path": "Video_0091/imgs", "startFrame": 1, "endFrame": 176, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0091/groundtruth_rect.txt", },

            {"name": "Video_0092", "path": "Video_0092/imgs", "startFrame": 1, "endFrame": 181, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0092/groundtruth_rect.txt", },

            {"name": "Video_0093", "path": "Video_0093/imgs", "startFrame": 1, "endFrame": 318, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0093/groundtruth_rect.txt", },

            {"name": "Video_0094", "path": "Video_0094/imgs", "startFrame": 1, "endFrame": 270, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0094/groundtruth_rect.txt", },

            {"name": "Video_0095", "path": "Video_0095/imgs", "startFrame": 1, "endFrame": 465, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0095/groundtruth_rect.txt", },

            {"name": "Video_0096", "path": "Video_0096/imgs", "startFrame": 1, "endFrame": 329, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0096/groundtruth_rect.txt", },

            {"name": "Video_0097", "path": "Video_0097/imgs", "startFrame": 1, "endFrame": 330, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0097/groundtruth_rect.txt", },

            {"name": "Video_0098", "path": "Video_0098/imgs", "startFrame": 1, "endFrame": 487, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0098/groundtruth_rect.txt", },

            {"name": "Video_0099", "path": "Video_0099/imgs", "startFrame": 1, "endFrame": 130, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0099/groundtruth_rect.txt", },

            {"name": "Video_0100", "path": "Video_0100/imgs", "startFrame": 1, "endFrame": 125, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0100/groundtruth_rect.txt", },

            {"name": "Video_0101", "path": "Video_0101/imgs", "startFrame": 1, "endFrame": 120, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0101/groundtruth_rect.txt", },

            {"name": "Video_0102", "path": "Video_0102/imgs", "startFrame": 1, "endFrame": 228, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0102/groundtruth_rect.txt", },

            {"name": "Video_0103", "path": "Video_0103/imgs", "startFrame": 1, "endFrame": 213, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0103/groundtruth_rect.txt", },

            {"name": "Video_0104", "path": "Video_0104/imgs", "startFrame": 1, "endFrame": 344, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0104/groundtruth_rect.txt", },

            {"name": "Video_0105", "path": "Video_0105/imgs", "startFrame": 1, "endFrame": 232, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0105/groundtruth_rect.txt", },

            {"name": "Video_0106", "path": "Video_0106/imgs", "startFrame": 1, "endFrame": 361, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0106/groundtruth_rect.txt", },

            {"name": "Video_0107", "path": "Video_0107/imgs", "startFrame": 1, "endFrame": 269, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0107/groundtruth_rect.txt", },

            {"name": "Video_0108", "path": "Video_0108/imgs", "startFrame": 1, "endFrame": 170, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0108/groundtruth_rect.txt", },

            {"name": "Video_0109", "path": "Video_0109/imgs", "startFrame": 1, "endFrame": 301, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0109/groundtruth_rect.txt", },

            {"name": "Video_0110", "path": "Video_0110/imgs", "startFrame": 1, "endFrame": 237, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0110/groundtruth_rect.txt", },

            {"name": "Video_0111", "path": "Video_0111/imgs", "startFrame": 1, "endFrame": 415, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0111/groundtruth_rect.txt", },

            {"name": "Video_0112", "path": "Video_0112/imgs", "startFrame": 1, "endFrame": 390, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0112/groundtruth_rect.txt", },

            {"name": "Video_0113", "path": "Video_0113/imgs", "startFrame": 1, "endFrame": 386, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0113/groundtruth_rect.txt", },

            {"name": "Video_0114", "path": "Video_0114/imgs", "startFrame": 1, "endFrame": 284, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0114/groundtruth_rect.txt", },

            {"name": "Video_0115", "path": "Video_0115/imgs", "startFrame": 1, "endFrame": 228, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0115/groundtruth_rect.txt", },

            {"name": "Video_0116", "path": "Video_0116/imgs", "startFrame": 1, "endFrame": 228, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0116/groundtruth_rect.txt", },

            {"name": "Video_0117", "path": "Video_0117/imgs", "startFrame": 1, "endFrame": 213, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0117/groundtruth_rect.txt", },

            {"name": "Video_0118", "path": "Video_0118/imgs", "startFrame": 1, "endFrame": 232, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0118/groundtruth_rect.txt", },

            {"name": "Video_0119", "path": "Video_0119/imgs", "startFrame": 1, "endFrame": 232, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0119/groundtruth_rect.txt", },

            {"name": "Video_0120", "path": "Video_0120/imgs", "startFrame": 1, "endFrame": 232, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0120/groundtruth_rect.txt", },

            {"name": "Video_0121", "path": "Video_0121/imgs", "startFrame": 1, "endFrame": 237, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0121/groundtruth_rect.txt", },

            {"name": "Video_0122", "path": "Video_0122/imgs", "startFrame": 1, "endFrame": 429, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0122/groundtruth_rect.txt", },

            {"name": "Video_0123", "path": "Video_0123/imgs", "startFrame": 1, "endFrame": 220, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0123/groundtruth_rect.txt", },

            {"name": "Video_0124", "path": "Video_0124/imgs", "startFrame": 1, "endFrame": 451, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0124/groundtruth_rect.txt", },

            {"name": "Video_0125", "path": "Video_0125/imgs", "startFrame": 1, "endFrame": 451, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0125/groundtruth_rect.txt", },

            {"name": "Video_0126", "path": "Video_0126/imgs", "startFrame": 1, "endFrame": 316, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0126/groundtruth_rect.txt", },

            {"name": "Video_0127", "path": "Video_0127/imgs", "startFrame": 1, "endFrame": 316, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0127/groundtruth_rect.txt", },

            {"name": "Video_0128", "path": "Video_0128/imgs", "startFrame": 1, "endFrame": 416, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0128/groundtruth_rect.txt", },

            {"name": "Video_0129", "path": "Video_0129/imgs", "startFrame": 1, "endFrame": 362, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0129/groundtruth_rect.txt", },

            {"name": "Video_0130", "path": "Video_0130/imgs", "startFrame": 1, "endFrame": 334, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0130/groundtruth_rect.txt", },

            {"name": "Video_0131", "path": "Video_0131/imgs", "startFrame": 1, "endFrame": 334, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0131/groundtruth_rect.txt", },

            {"name": "Video_0132", "path": "Video_0132/imgs", "startFrame": 1, "endFrame": 503, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0132/groundtruth_rect.txt", },

            {"name": "Video_0133", "path": "Video_0133/imgs", "startFrame": 1, "endFrame": 363, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0133/groundtruth_rect.txt", },

            {"name": "Video_0134", "path": "Video_0134/imgs", "startFrame": 1, "endFrame": 338, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0134/groundtruth_rect.txt", },

            {"name": "Video_0135", "path": "Video_0135/imgs", "startFrame": 1, "endFrame": 325, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0135/groundtruth_rect.txt", },

            {"name": "Video_0136", "path": "Video_0136/imgs", "startFrame": 1, "endFrame": 493, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0136/groundtruth_rect.txt", },

            {"name": "Video_0137", "path": "Video_0137/imgs", "startFrame": 1, "endFrame": 391, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0137/groundtruth_rect.txt", },

            {"name": "Video_0138", "path": "Video_0138/imgs", "startFrame": 1, "endFrame": 391, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0138/groundtruth_rect.txt", },

            {"name": "Video_0139", "path": "Video_0139/imgs", "startFrame": 1, "endFrame": 386, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0139/groundtruth_rect.txt", },

            {"name": "Video_0140", "path": "Video_0140/imgs", "startFrame": 1, "endFrame": 386, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0140/groundtruth_rect.txt", },

            {"name": "Video_0141", "path": "Video_0141/imgs", "startFrame": 1, "endFrame": 379, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0141/groundtruth_rect.txt", },

            {"name": "Video_0142", "path": "Video_0142/imgs", "startFrame": 1, "endFrame": 363, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0142/groundtruth_rect.txt", },

            {"name": "Video_0143", "path": "Video_0143/imgs", "startFrame": 1, "endFrame": 471, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0143/groundtruth_rect.txt", },

            {"name": "Video_0144", "path": "Video_0144/imgs", "startFrame": 1, "endFrame": 270, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0144/groundtruth_rect.txt", },

            {"name": "Video_0145", "path": "Video_0145/imgs", "startFrame": 1, "endFrame": 287, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0145/groundtruth_rect.txt", },

            {"name": "Video_0151", "path": "Video_0151/imgs", "startFrame": 1, "endFrame": 266, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0151/groundtruth_rect.txt", },

            {"name": "Video_0152", "path": "Video_0152/imgs", "startFrame": 1, "endFrame": 292, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0152/groundtruth_rect.txt", },

            {"name": "Video_0153", "path": "Video_0153/imgs", "startFrame": 1, "endFrame": 387, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0153/groundtruth_rect.txt", },

            {"name": "Video_0154", "path": "Video_0154/imgs", "startFrame": 1, "endFrame": 167, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0154/groundtruth_rect.txt", },

            {"name": "Video_0155", "path": "Video_0155/imgs", "startFrame": 1, "endFrame": 351, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0155/groundtruth_rect.txt", },

            {"name": "Video_0156", "path": "Video_0156/imgs", "startFrame": 1, "endFrame": 213, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0156/groundtruth_rect.txt", },
            #####
            {"name": "Video_0157", "path": "Video_0157/imgs", "startFrame": 1, "endFrame": 462, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0157/groundtruth_rect.txt", },

            {"name": "Video_0158", "path": "Video_0158/imgs", "startFrame": 1, "endFrame": 266, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0158/groundtruth_rect.txt", },

            {"name": "Video_0159", "path": "Video_0159/imgs", "startFrame": 1, "endFrame": 305, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0159/groundtruth_rect.txt", },

            {"name": "Video_0160", "path": "Video_0160/imgs", "startFrame": 1, "endFrame": 370, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0160/groundtruth_rect.txt", },

            {"name": "Video_0161", "path": "Video_0161/imgs", "startFrame": 1, "endFrame": 447, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0161/groundtruth_rect.txt", },

            {"name": "Video_0162", "path": "Video_0162/imgs", "startFrame": 1, "endFrame": 194, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0162/groundtruth_rect.txt", },

            {"name": "Video_0163", "path": "Video_0163/imgs", "startFrame": 1, "endFrame": 155, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0163/groundtruth_rect.txt", },

            {"name": "Video_0164", "path": "Video_0164/imgs", "startFrame": 1, "endFrame": 408, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0164/groundtruth_rect.txt", },

            {"name": "Video_0165", "path": "Video_0165/imgs", "startFrame": 1, "endFrame": 379, "nz": 4, "ext": "jpg",
             "anno_path": "Video_0165/groundtruth_rect.txt", },
        ]

        return sequence_info_list
