import os
from lib.utils.load import load_yaml
from lib.test.utils import TrackerParams
from lib.test.evaluation.environment import env_settings


def parameters(yaml_name: str, env_num: int):
    params = TrackerParams()
    params.env_num = env_num
    prj_dir = env_settings(env_num).prj_dir
    save_dir = env_settings(env_num).save_dir
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/lightfc/%s.yaml' % yaml_name)
    params.cfg = load_yaml(yaml_file)
    print("test config: ", params.cfg)
    params.tracker_param = yaml_name

    # template and search region
    params.template_factor = params.cfg.TEST.TEMPLATE_FACTOR
    params.template_size = params.cfg.TEST.TEMPLATE_SIZE
    params.search_factor = params.cfg.TEST.SEARCH_FACTOR
    params.search_size = params.cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = os.path.join(save_dir, "checkpoints/train/lightfc/%s/lightfc_ep%04d.pth.tar" %
                                     (yaml_name, params.cfg.TEST.EPOCH))

    params.save_all_boxes = False

    return params
