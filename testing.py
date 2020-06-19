import copy as cp
import time
import pickle
from scipy.special import expit
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

from Utils.eval_helper import *
from Utils.iohelper import *
from Detector.eval_GANG import *
from Detector.eval_Fraudar import *
from Detector.eval_SpEagle import *
from Detector.eval_fBox import *
from nash_detect import compute_re, load_fake_reviews, remove_topk_reviews, e_greedy_sample


"""
	Testing all detectors (Figure 4 in paper)
"""


def testing_detectors(upg, pug, added_reviews, detectors_q, top_k_per, detector='GANG'):
	"""
	Run detectors on all reviews plus added fake reviews.
	Return the aggregated review posterior beliefs and top_k suspicious reviews
	"""

	spammer_ids = []
	for review in added_reviews:
		if review[0] not in spammer_ids:
			spammer_ids.append(review[0])

	u_p_graph, p_u_graph = cp.deepcopy(upg), cp.deepcopy(pug)

	new_priors, u_p_graph, p_u_graph, user_ground_truth, review_ground_truth, _ = add_adversarial_review(
		u_p_graph, p_u_graph, added_reviews)

	if detector == 'GANG':
		print('Run {} ...'.format(detector))
		# get posteriors from GANG
		gang_model, _ = runGANG(new_priors, u_p_graph, p_u_graph, user_ground_truth)
		gang_ubelief, _, rbelief = gang_model.classify()
	elif detector == 'SpEagle':
		print('Run {} ...'.format(detector))
		# get posteriors from SpEagle
		speagle_model = runSpEagle(new_priors, u_p_graph)
		speagle_ubelief, rbelief, _ = speagle_model.classify()
	elif detector == 'Fraudar':
		print('Run {} ...'.format(detector))
		# get posteriors from Fraudar
		fraudar_ubelief, rbelief = runFraudar(new_priors, u_p_graph)
	elif detector == 'fBox':
		print('Run {} ...'.format(detector))
		# get posteriors from fBox
		fbox_ubelief, rbelief = runfBox(new_priors, u_p_graph)
	elif detector == 'Prior':
		print('Run {} ...'.format(detector))
		# get posteriors from Prior
		prior_ubelief, rbelief = new_priors[0], new_priors[1]
	else:
		# run Nash-Detect or Equal-Weights
		print('Run {} ...'.format(detector))
		gang_model, _ = runGANG(new_priors, u_p_graph, p_u_graph, user_ground_truth)
		gang_ubelief, _, gang_rbelief = gang_model.classify()
		speagle_model = runSpEagle(new_priors, u_p_graph)
		speagle_ubelief, speagle_rbelief, _ = speagle_model.classify()
		fraudar_ubelief, fraudar_rbelief = runFraudar(new_priors, u_p_graph)
		fbox_ubelief, fbox_rbelief = runfBox(new_priors, u_p_graph)
		prior_ubelief, prior_rbelief = new_priors[0], new_priors[1]

	print('Compute Posterior ...')
	if detector == 'Nash-Detect' or detector == 'Equal-Weights':
		# normalize the output
		speagle_rbelief, fraudar_rbelief, fbox_rbelief, prior_rbelief = scale_value(speagle_rbelief), scale_value(
			fraudar_rbelief), scale_value(fbox_rbelief), scale_value(prior_rbelief)

		# weight of each detector
		gang_q, speagle_q, fraudar_q, fbox_q, prior_q = detectors_q['GANG'], detectors_q['SpEagle'], detectors_q['Fraudar'], \
													detectors_q['fBox'], detectors_q['Prior']
		# compute aggregated review posteriors
		r_spam_beliefs = {}

		if detector == 'Nash-Detect':
			for r in gang_rbelief.keys():
				accu_spam_belief = 0.0
				accu_spam_belief += gang_q * gang_rbelief[r]
				accu_spam_belief += speagle_q * speagle_rbelief[r]
				accu_spam_belief += fraudar_q * fraudar_rbelief[r]
				accu_spam_belief += fbox_q * fbox_rbelief[r]
				accu_spam_belief += prior_q * prior_rbelief[r]

				r_spam_beliefs[r] = expit(accu_spam_belief)
		else:
			for r in gang_rbelief.keys():
				accu_spam_belief = gang_rbelief[r] + speagle_rbelief[r] + fraudar_rbelief[r] + fbox_rbelief[r] + prior_rbelief[r]
				r_spam_beliefs[r] = expit(accu_spam_belief)

	else:
		r_spam_beliefs = scale_value(rbelief)

	# rank the top_k suspicious reviews
	ranked_rbeliefs = [(review, r_spam_beliefs[review]) for review in r_spam_beliefs.keys()]
	ranked_rbeliefs = sorted(ranked_rbeliefs, reverse=True, key=lambda x: x[1])
	top_k = int(len(r_spam_beliefs) * top_k_per)
	top_k_reviews = [review[0] for review in ranked_rbeliefs[:top_k]]

	# remove false positives
	for review in top_k_reviews:
		if review not in added_reviews:
			top_k_reviews.remove(review)

	print('top k is {}'.format(len(top_k_reviews)))

	return r_spam_beliefs, top_k_reviews, p_u_graph


if __name__ == '__main__':

	# load metadata and attack defense data
	dataset_name = 'YelpChi'  # YelpChi, YelpNYC, YelpZip
	prefix = 'Yelp_Dataset/' + dataset_name + '/'
	elite = 10  # elite threshold
	top_k = 0.01  # filtering threshold for detector
	epsilon = 0.1  # epsilon-greedy sampling parameter
	lr1_dict = {'YelpChi': 30, 'YelpNYC': 40, 'YelpZip': 48}
	lr1 = lr1_dict[dataset_name]  # learning rate for detector importance
	lr2 = 0.01  # learning rate for spamming mixture parameter
	iters = 10  # running iterations
	testing_detector = 'Equal-Weights'  # Equal-Weights, Nash-Detect, Fraudar, SpEagle, GANG, Prior, fBox,

	attacks = ['IncBP', 'IncDS', 'IncPR', 'Random', 'Singleton']
	detectors = ['GANG', 'Prior', 'SpEagle', 'fBox', 'Fraudar']

	setting1 = 'Testing/' + dataset_name + '/IncBP.pickle'
	setting2 = 'Testing/' + dataset_name + '/IncDS.pickle'
	setting3 = 'Testing/' + dataset_name + '/Random.pickle'
	setting4 = 'Testing/' + dataset_name + '/IncPR.pickle'
	setting5 = 'Testing/' + dataset_name + '/Singleton.pickle'

	paths = {'IncBP': setting1, 'IncDS': setting2, 'Random': setting3, 'IncPR': setting4,
			 'Singleton': setting5}

	metadata_filename = prefix + 'metadata.gz'
	user_product_graph, prod_user_graph = read_graph_data(metadata_filename)

	with open(setting1, 'rb') as f:
		evasions = pickle.load(f)
		targets = evasions[1]

	# load fake reviews for each attack to each target item
	item_attack_review = load_fake_reviews(attacks, targets, paths)

	# select elite accounts
	elite_accounts = select_elite(user_product_graph, threshold=elite)

	with open(paths['IncDS'], 'rb') as f:
		evasions = pickle.load(f)

	# initialize P and Q
	attacks_p = {}
	optimal_q = {'YelpChi': {'GANG': 1.61, 'Prior': 0.60, 'SpEagle': 0.21, 'fBox': 0.40, 'Fraudar': 2.22},
				 'YelpNYC': {'GANG': 2.47, 'Prior': 0.69, 'SpEagle': 0.21, 'fBox': 0.43, 'Fraudar': 2.00},
				 'YelpZip': {'GANG': 2.58, 'Prior': 0.66, 'SpEagle': 0.20, 'fBox': 0.36, 'Fraudar': 1.94}}
	detectors_q = optimal_q[dataset_name]

	# uniformly initialization
	for a in attacks:
		attacks_p[a] = 1 / len(attacks)

	# initialize loggers
	posted_reviews = {}
	item_attack_mapping = {}
	attack_log = {a: [] for a in attacks_p.keys()}
	pe_log, return_log = [], []
	init_time = time.time()

	# start attacks and defenses
	for i in range(iters):
		print('run detector {} on dataset {}, iteration is {}.'.format(testing_detector, dataset_name, i))
		posted_reviews[i] = []
		item_attack_mapping[i] = {}

		# generate fake reviews for each item
		for item in targets:
			attack = e_greedy_sample(attacks_p, epsilon)
			item_attack_mapping[i][item] = attack
			reviews = item_attack_review[item][attack]
			posted_reviews[i] += reviews

		# run all detectors on all reviews and added reviews
		r_spam_beliefs, top_k_reviews, new_prod_user_graph = testing_detectors(user_product_graph, prod_user_graph, posted_reviews[i], detectors_q, top_k, detector=testing_detector)

		# remove the top_k reviews and calculate the practical effect
		print('Compute original Revenue ...')
		ori_RI, ori_ERI, ori_Revenue = compute_re(prod_user_graph, targets, elite_accounts)
		remain_product_user_graph = remove_topk_reviews(new_prod_user_graph, top_k_reviews)

		print('Compute new Revenue ...')
		new_RI, new_ERI, new_Revenue = compute_re(remain_product_user_graph, targets, elite_accounts)

		# calculate total practical effect
		total_pe = sum([new_Revenue[t] - ori_Revenue[t] for t in targets])

		print('Total PE is {}'.format(total_pe))

		# logging
		pe_log.append(total_pe)

		new_time = time.time()
		print('\nTime cost is for iteration {} is {}.\n'.format(i, new_time - init_time))
		init_time = new_time

		print(pe_log)
		print('Avg PE is {}'.format(sum(pe_log) / len(pe_log)))
		print('Max PE is {}'.format(max(pe_log)))
		print('Min PE is {}'.format(min(pe_log)))


