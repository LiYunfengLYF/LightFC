import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class DTBDataset(BaseDataset):
    """ DTB70
    """

    def __init__(self, env_num):
        super().__init__(env_num)
        self.base_path = self.env_settings.dtb_path
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
        ground_truth_rect = load_text(str(anno_path), delimiter=(',', None), dtype=np.float64, backend='numpy')

        return Sequence(sequence_info['name'], frames, 'dtb', ground_truth_rect[init_omit:, :],)

    def __len__(self):
        return len(self.sequence_info_list)

    def _get_sequence_info_list(self):
        sequence_info_list = [
            {"name": "Animal1", "path": "Animal1/img", "startFrame": 1, "endFrame": 147, "nz": 5, "ext": "jpg",
             "anno_path": "Animal1/groundtruth_rect.txt"},

            {"name": "Animal2", "path": "Animal2/img", "startFrame": 1, "endFrame": 149, "nz": 5, "ext": "jpg",
             "anno_path": "Animal2/groundtruth_rect.txt"},

            {"name": "Animal3", "path": "Animal3/img", "startFrame": 1, "endFrame": 309, "nz": 5, "ext": "jpg",
             "anno_path": "Animal3/groundtruth_rect.txt"},

            {"name": "Animal4", "path": "Animal4/img", "startFrame": 1, "endFrame": 167, "nz": 5, "ext": "jpg",
             "anno_path": "Animal4/groundtruth_rect.txt"},

            {"name": "Basketball", "path": "Basketball/img", "startFrame": 1, "endFrame": 427, "nz": 5, "ext": "jpg",
             "anno_path": "Basketball/groundtruth_rect.txt"},

            {"name": "BMX2", "path": "BMX2/img", "startFrame": 1, "endFrame": 160, "nz": 5, "ext": "jpg",
             "anno_path": "BMX2/groundtruth_rect.txt"},

            {"name": "BMX3", "path": "BMX3/img", "startFrame": 1, "endFrame": 121, "nz": 5, "ext": "jpg",
             "anno_path": "BMX3/groundtruth_rect.txt"},

            {"name": "BMX4", "path": "BMX4/img", "startFrame": 1, "endFrame": 364, "nz": 5, "ext": "jpg",
             "anno_path": "BMX4/groundtruth_rect.txt"},

            {"name": "BMX5", "path": "BMX5/img", "startFrame": 1, "endFrame": 277, "nz": 5, "ext": "jpg",
             "anno_path": "BMX5/groundtruth_rect.txt"},

            {"name": "Car2", "path": "Car2/img", "startFrame": 1, "endFrame": 74, "nz": 5, "ext": "jpg",
             "anno_path": "Car2/groundtruth_rect.txt"},

            {"name": "Car4", "path": "Car4/img", "startFrame": 1, "endFrame": 217, "nz": 5, "ext": "jpg",
             "anno_path": "Car4/groundtruth_rect.txt"},

            {"name": "Car5", "path": "Car5/img", "startFrame": 1, "endFrame": 84, "nz": 5, "ext": "jpg",
             "anno_path": "Car5/groundtruth_rect.txt"},

            {"name": "Car6", "path": "Car6/img", "startFrame": 1, "endFrame": 381, "nz": 5, "ext": "jpg",
             "anno_path": "Car6/groundtruth_rect.txt"},

            {"name": "Car8", "path": "Car8/img", "startFrame": 1, "endFrame": 319, "nz": 5, "ext": "jpg",
             "anno_path": "Car8/groundtruth_rect.txt"},

            {"name": "ChasingDrones", "path": "ChasingDrones/img", "startFrame": 1, "endFrame": 212, "nz": 5,
             "ext": "jpg", "anno_path": "ChasingDrones/groundtruth_rect.txt"},

            {"name": "Girl1", "path": "Girl1/img", "startFrame": 1, "endFrame": 218, "nz": 5,
             "ext": "jpg", "anno_path": "Girl1/groundtruth_rect.txt"},

            {"name": "Girl2", "path": "Girl2/img", "startFrame": 1, "endFrame": 626, "nz": 5,
             "ext": "jpg", "anno_path": "Girl2/groundtruth_rect.txt"},

            {"name": "Gull1", "path": "Gull1/img", "startFrame": 1, "endFrame": 120, "nz": 5,
             "ext": "jpg", "anno_path": "Gull1/groundtruth_rect.txt"},

            {"name": "Gull2", "path": "Gull2/img", "startFrame": 1, "endFrame": 237, "nz": 5,
             "ext": "jpg", "anno_path": "Gull2/groundtruth_rect.txt"},

            {"name": "Horse1", "path": "Horse1/img", "startFrame": 1, "endFrame": 149, "nz": 5,
             "ext": "jpg", "anno_path": "Horse1/groundtruth_rect.txt"},

            {"name": "Horse2", "path": "Horse2/img", "startFrame": 1, "endFrame": 126, "nz": 5,
             "ext": "jpg", "anno_path": "Horse2/groundtruth_rect.txt"},

            {"name": "Kiting", "path": "Kiting/img", "startFrame": 1, "endFrame": 129, "nz": 5,
             "ext": "jpg", "anno_path": "Kiting/groundtruth_rect.txt"},

            {"name": "ManRunning1", "path": "ManRunning1/img", "startFrame": 1, "endFrame": 619, "nz": 5,
             "ext": "jpg", "anno_path": "ManRunning1/groundtruth_rect.txt"},

            {"name": "ManRunning2", "path": "ManRunning2/img", "startFrame": 1, "endFrame": 260, "nz": 5,
             "ext": "jpg", "anno_path": "ManRunning2/groundtruth_rect.txt"},

            {"name": "Motor1", "path": "Motor1/img", "startFrame": 1, "endFrame": 78, "nz": 5,
             "ext": "jpg", "anno_path": "Motor1/groundtruth_rect.txt"},

            {"name": "Motor2", "path": "Motor2/img", "startFrame": 1, "endFrame": 68, "nz": 5,
             "ext": "jpg", "anno_path": "Motor2/groundtruth_rect.txt"},

            {"name": "MountainBike1", "path": "MountainBike1/img", "startFrame": 1, "endFrame": 109, "nz": 5,
             "ext": "jpg", "anno_path": "MountainBike1/groundtruth_rect.txt"},

            {"name": "MountainBike5", "path": "MountainBike5/img", "startFrame": 1, "endFrame": 107, "nz": 5,
             "ext": "jpg", "anno_path": "MountainBike5/groundtruth_rect.txt"},

            {"name": "MountainBike6", "path": "MountainBike6/img", "startFrame": 1, "endFrame": 81, "nz": 5,
             "ext": "jpg", "anno_path": "MountainBike6/groundtruth_rect.txt"},

            {"name": "Paragliding3", "path": "Paragliding3/img", "startFrame": 1, "endFrame": 211, "nz": 5,
             "ext": "jpg", "anno_path": "Paragliding3/groundtruth_rect.txt"},

            {"name": "Paragliding5", "path": "Paragliding5/img", "startFrame": 1, "endFrame": 220, "nz": 5,
             "ext": "jpg", "anno_path": "Paragliding5/groundtruth_rect.txt"},

            {"name": "RaceCar", "path": "RaceCar/img", "startFrame": 1, "endFrame": 80, "nz": 5,
             "ext": "jpg", "anno_path": "RaceCar/groundtruth_rect.txt"},

            {"name": "RaceCar1", "path": "RaceCar1/img", "startFrame": 1, "endFrame": 249, "nz": 5,
             "ext": "jpg", "anno_path": "RaceCar1/groundtruth_rect.txt"},

            {"name": "RcCar3", "path": "RcCar3/img", "startFrame": 1, "endFrame": 202, "nz": 5,
             "ext": "jpg", "anno_path": "RcCar3/groundtruth_rect.txt"},

            {"name": "RcCar4", "path": "RcCar4/img", "startFrame": 1, "endFrame": 699, "nz": 5,
             "ext": "jpg", "anno_path": "RcCar4/groundtruth_rect.txt"},

            {"name": "RcCar5", "path": "RcCar5/img", "startFrame": 1, "endFrame": 330, "nz": 5,
             "ext": "jpg", "anno_path": "RcCar5/groundtruth_rect.txt"},

            {"name": "RcCar6", "path": "RcCar6/img", "startFrame": 1, "endFrame": 210, "nz": 5,
             "ext": "jpg", "anno_path": "RcCar6/groundtruth_rect.txt"},

            {"name": "RcCar7", "path": "RcCar7/img", "startFrame": 1, "endFrame": 287, "nz": 5,
             "ext": "jpg", "anno_path": "RcCar7/groundtruth_rect.txt"},

            {"name": "RcCar8", "path": "RcCar8/img", "startFrame": 1, "endFrame": 211, "nz": 5,
             "ext": "jpg", "anno_path": "RcCar8/groundtruth_rect.txt"},

            {"name": "RcCar9", "path": "RcCar9/img", "startFrame": 1, "endFrame": 208, "nz": 5,
             "ext": "jpg", "anno_path": "RcCar9/groundtruth_rect.txt"},

            {"name": "Sheep1", "path": "Sheep1/img", "startFrame": 1, "endFrame": 391, "nz": 5,
             "ext": "jpg", "anno_path": "Sheep1/groundtruth_rect.txt"},

            {"name": "Sheep2", "path": "Sheep2/img", "startFrame": 1, "endFrame": 251, "nz": 5,
             "ext": "jpg", "anno_path": "Sheep2/groundtruth_rect.txt"},

            {"name": "SkateBoarding4", "path": "SkateBoarding4/img", "startFrame": 1, "endFrame": 201, "nz": 5,
             "ext": "jpg", "anno_path": "SkateBoarding4/groundtruth_rect.txt"},

            {"name": "Skiing1", "path": "Skiing1/img", "startFrame": 1, "endFrame": 87, "nz": 5,
             "ext": "jpg", "anno_path": "Skiing1/groundtruth_rect.txt"},

            {"name": "Skiing2", "path": "Skiing2/img", "startFrame": 1, "endFrame": 178, "nz": 5,
             "ext": "jpg", "anno_path": "Skiing2/groundtruth_rect.txt"},

            {"name": "SnowBoarding2", "path": "SnowBoarding2/img", "startFrame": 1, "endFrame": 117, "nz": 5,
             "ext": "jpg", "anno_path": "SnowBoarding2/groundtruth_rect.txt"},

            {"name": "SnowBoarding4", "path": "SnowBoarding4/img", "startFrame": 1, "endFrame": 109, "nz": 5,
             "ext": "jpg", "anno_path": "SnowBoarding4/groundtruth_rect.txt"},

            {"name": "SnowBoarding6", "path": "SnowBoarding6/img", "startFrame": 1, "endFrame": 100, "nz": 5,
             "ext": "jpg", "anno_path": "SnowBoarding6/groundtruth_rect.txt"},

            {"name": "Soccer1", "path": "Soccer1/img", "startFrame": 1, "endFrame": 613, "nz": 5,
             "ext": "jpg", "anno_path": "Soccer1/groundtruth_rect.txt"},

            {"name": "Soccer2", "path": "Soccer2/img", "startFrame": 1, "endFrame": 233, "nz": 5,
             "ext": "jpg", "anno_path": "Soccer2/groundtruth_rect.txt"},

            {"name": "SpeedCar2", "path": "SpeedCar2/img", "startFrame": 1, "endFrame": 308, "nz": 5,
             "ext": "jpg", "anno_path": "SpeedCar2/groundtruth_rect.txt"},

            {"name": "SpeedCar4", "path": "SpeedCar4/img", "startFrame": 1, "endFrame": 164, "nz": 5,
             "ext": "jpg", "anno_path": "SpeedCar4/groundtruth_rect.txt"},

            {"name": "StreetBasketball1", "path": "StreetBasketball1/img", "startFrame": 1, "endFrame": 241, "nz": 5,
             "ext": "jpg", "anno_path": "StreetBasketball1/groundtruth_rect.txt"},

            {"name": "StreetBasketball2", "path": "StreetBasketball2/img", "startFrame": 1, "endFrame": 114, "nz": 5,
             "ext": "jpg", "anno_path": "StreetBasketball2/groundtruth_rect.txt"},

            {"name": "StreetBasketball3", "path": "StreetBasketball3/img", "startFrame": 1, "endFrame": 501, "nz": 5,
             "ext": "jpg", "anno_path": "StreetBasketball3/groundtruth_rect.txt"},

            {"name": "SUP2", "path": "SUP2/img", "startFrame": 1, "endFrame": 133, "nz": 5,
             "ext": "jpg", "anno_path": "SUP2/groundtruth_rect.txt"},

            {"name": "SUP4", "path": "SUP4/img", "startFrame": 1, "endFrame": 295, "nz": 5,
             "ext": "jpg", "anno_path": "SUP4/groundtruth_rect.txt"},

            {"name": "SUP5", "path": "SUP5/img", "startFrame": 1, "endFrame": 222, "nz": 5,
             "ext": "jpg", "anno_path": "SUP5/groundtruth_rect.txt"},

            {"name": "Surfing03", "path": "Surfing03/img", "startFrame": 1, "endFrame": 117, "nz": 5,
             "ext": "jpg", "anno_path": "Surfing03/groundtruth_rect.txt"},

            {"name": "Surfing04", "path": "Surfing04/img", "startFrame": 1, "endFrame": 158, "nz": 5,
             "ext": "jpg", "anno_path": "Surfing04/groundtruth_rect.txt"},

            {"name": "Surfing06", "path": "Surfing06/img", "startFrame": 1, "endFrame": 106, "nz": 5,
             "ext": "jpg", "anno_path": "Surfing06/groundtruth_rect.txt"},

            {"name": "Surfing10", "path": "Surfing10/img", "startFrame": 1, "endFrame": 165, "nz": 5,
             "ext": "jpg", "anno_path": "Surfing10/groundtruth_rect.txt"},

            {"name": "Surfing12", "path": "Surfing12/img", "startFrame": 1, "endFrame": 135, "nz": 5,
             "ext": "jpg", "anno_path": "Surfing12/groundtruth_rect.txt"},

            {"name": "Vaulting", "path": "Vaulting/img", "startFrame": 1, "endFrame": 171, "nz": 5,
             "ext": "jpg", "anno_path": "Vaulting/groundtruth_rect.txt"},

            {"name": "Wakeboarding1", "path": "Wakeboarding1/img", "startFrame": 1, "endFrame": 77, "nz": 5,
             "ext": "jpg", "anno_path": "Wakeboarding1/groundtruth_rect.txt"},

            {"name": "Wakeboarding2", "path": "Wakeboarding2/img", "startFrame": 1, "endFrame": 76, "nz": 5,
             "ext": "jpg", "anno_path": "Wakeboarding2/groundtruth_rect.txt"},

            {"name": "Walking", "path": "Walking/img", "startFrame": 1, "endFrame": 395, "nz": 5,
             "ext": "jpg", "anno_path": "Walking/groundtruth_rect.txt"},

            {"name": "Yacht2", "path": "Yacht2/img", "startFrame": 1, "endFrame": 239, "nz": 5,
             "ext": "jpg", "anno_path": "Yacht2/groundtruth_rect.txt"},

            {"name": "Yacht4", "path": "Yacht4/img", "startFrame": 1, "endFrame": 461, "nz": 5,
             "ext": "jpg", "anno_path": "Yacht4/groundtruth_rect.txt"},

            {"name": "Zebra", "path": "Zebra/img", "startFrame": 1, "endFrame": 177, "nz": 5,
             "ext": "jpg", "anno_path": "Zebra/groundtruth_rect.txt"},
        ]

        return sequence_info_list
