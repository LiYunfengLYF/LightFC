from lib.test.evaluation.environment import EnvSettings


def local_env_settings(env_num):
    settings = EnvSettings()

    settings.davis_dir = r''
    settings.got10k_lmdb_path = r''
    settings.got10k_path = r''
    settings.got_packed_results_path = r''
    settings.got_reports_path = r''
    settings.itb_path = r''
    settings.lasot_extension_subset_path = ''
    settings.lasot_lmdb_path = r''
    settings.lasot_path = '/media/liyunfeng/CV2/data/sot/lasot'
    settings.network_path = r''
    settings.nfs_path = r''
    settings.otb_path = r'/media/liyunfeng/CV2/data/sot/otb'
    settings.dtb_path = r''
    settings.prj_dir = r'/home/liyunfeng/code/project2/LightFC'
    settings.result_plot_path = r'/home/liyunfeng/code/project2/LightFC/output/test/result_plots'
    # Where to store tracking results
    settings.results_path = r'/home/liyunfeng/code/project2/LightFC/output/test/tracking_results'
    settings.save_dir = r'/home/liyunfeng/code/project2/LightFC/output'
    settings.segmentation_path = r'/home/liyunfeng/code/project2/LightFC/output/test/segmentation_results'
    settings.tc128_path = r'/media/liyunfeng/CV2/data/sot/tc128'
    settings.tn_packed_results_path = r''
    settings.tnl2k_path = r'/media/liyunfeng/CV2/data/sot/tnl2k/test'
    settings.tpl_path = r''
    settings.trackingnet_path = r''
    settings.uav_path = r'/media/liyunfeng/CV2/data/uav/uav123'
    settings.vot18_path = r''
    settings.vot22_path = r''
    settings.vot_path = r''
    settings.youtubevos_dir = r''
    settings.uot_path = r'/media/liyunfeng/CV2/data/uot/uot100'
    settings.utb_path = r'/media/liyunfeng/CV2/data/uot/utb180'

    return settings
