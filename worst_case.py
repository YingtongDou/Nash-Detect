import copy as cp
import pickle
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

from Utils.eval_helper import *
from Utils.iohelper import *
from Utils.yelpFeatureExtraction import *
from Detector.eval_GANG import runGANG
from Detector.eval_Fraudar import runFraudar
from Detector.eval_SpEagle import runSpEagle
from Detector.eval_fBox import runfBox
from nash_detect import compute_re, remove_topk_reviews


"""
	Computing worst-case performance (Table 2 and Figure 3(d) dash lines in paper)
"""


if __name__ == '__main__':

	# load metadata and attack defense data
	dataset_name = 'YelpChi'  # YelpChi, YelpNYC, YelpZip
	prefix = 'Yelp_Dataset/' + dataset_name + '/'
	elite = 10  # elite threshold
	top_k = 0.01  # filtering threshold for detector
	mode = 'Training/'

	attacks = ['IncBP', 'IncDS', 'IncPR', 'Random', 'Singleton']
	detectors = ['GANG', 'Prior', 'SpEagle', 'fBox', 'Fraudar']

	setting1 = mode + dataset_name + '/IncBP.pickle'
	setting2 = mode + dataset_name + '/IncDS.pickle'
	setting3 = mode + dataset_name + '/Random.pickle'
	setting4 = mode + dataset_name + '/IncPR.pickle'
	setting5 = mode + dataset_name + '/Singleton.pickle'

	paths = {'IncBP': setting1, 'IncDS': setting2, 'Random': setting3, 'IncPR': setting4,
			 'Singleton': setting5}

	topk_log = {a: {d: [] for d in detectors} for a in attacks}
	pe_log = {a: {d: 0 for d in detectors} for a in attacks}

	metadata_filename = prefix + 'metadata.gz'
	user_product_graph, prod_user_graph = read_graph_data(metadata_filename)

	# select elite accounts
	elite_accounts = select_elite(user_product_graph, threshold=elite)

	for attack, setting in paths.items():

		print("Run detectors on {} with {} attack now...".format(dataset_name, attack))

		with open(setting, 'rb') as f:
			evasions = pickle.load(f)
			account_ids = evasions[0]
			target_ids = evasions[1]
			new_edges = evasions[2]

		u_p_graph, p_u_graph = cp.deepcopy(user_product_graph), cp.deepcopy(prod_user_graph)
		new_priors, u_p_graph, p_u_graph, user_ground_truth, review_ground_truth, _ = add_adversarial_review(
			u_p_graph, p_u_graph, new_edges)

		print('Run GANG ...')
		# get posteriors from GANG
		gang_model, _ = runGANG(new_priors, u_p_graph, p_u_graph, user_ground_truth)
		gang_ubelief, _, gang_rbelief = gang_model.classify()

		print('Run SpEagle ...')
		# get posteriors from SpEagle
		speagle_model = runSpEagle(new_priors, u_p_graph)
		speagle_ubelief, speagle_rbelief, _ = speagle_model.classify()

		print('Run Fraudar ...')
		# get posteriors from Fraudar
		fraudar_ubelief, fraudar_rbelief = runFraudar(new_priors, u_p_graph)

		print('Run fBox ...')
		# get posteriors from fBox
		fbox_ubelief, fbox_rbelief = runfBox(new_priors, u_p_graph)

		print('Run Prior ...')
		# get posteriors from Prior
		prior_ubelief, prior_rbelief = new_priors[0], new_priors[1]

		# normalize the output
		speagle_rbelief, fraudar_rbelief, fbox_rbelief, prior_rbelief = scale_value(speagle_rbelief), scale_value(
			fraudar_rbelief), scale_value(fbox_rbelief), scale_value(prior_rbelief)

		detector_belief = {'GANG': gang_rbelief, 'SpEagle': speagle_rbelief, 'Fraudar': fraudar_rbelief, 'fBox': fbox_rbelief, 'Prior': prior_rbelief}

		# rank the top_k suspicious reviews
		for detector, belief in detector_belief.items():
			ranked_rbeliefs = [(review, belief[review]) for review in belief.keys()]
			ranked_rbeliefs = sorted(ranked_rbeliefs, reverse=True, key=lambda x: x[1])
			top_k_count = int(len(belief) * top_k)
			top_k_reviews = [review[0] for review in ranked_rbeliefs[:top_k_count]]

			topk_log[attack][detector] += top_k_reviews

		print('Compute PM ...')
		ori_RI, ori_ERI, ori_Revenue = compute_re(prod_user_graph, target_ids, elite_accounts)
		for detector, topk in topk_log[attack].items():
			remain_product_user_graph = remove_topk_reviews(p_u_graph, topk)
			new_RI, new_ERI, new_Revenue = compute_re(remain_product_user_graph, target_ids, elite_accounts)
			pe_log[attack][detector] = sum([new_Revenue[i] - ori_Revenue[i] for i in target_ids])

		print(pe_log)