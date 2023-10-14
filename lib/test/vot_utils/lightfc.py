from lib.test.vot_utils.lightfc_vot import run_vot_exp
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
run_vot_exp('lightfc', 'mobilnetv2_p_pwcorr_se_scf_sc_iab_sc_adj_concat_repn33_se_conv33_center_wiou', vis=False)