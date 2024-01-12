import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [8, 8]
env_num = 0
from lib.test.analysis.plot_results import print_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []
dataset_name = 'utb'


parameter_name = r'mobilnetv2_p_pwcorr_se_scf_sc_iab_sc_adj_concat_repn33_se_conv33_center_wiou'
trackers.extend(
    trackerlist(name='lightfc', parameter_name=parameter_name, dataset_name=dataset_name,
                run_ids=None, env_num=env_num, display_name=parameter_name))


dataset = get_dataset(dataset_name, env_num=env_num)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'),
              env_num=env_num)

