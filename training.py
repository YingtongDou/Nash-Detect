import copy as cp
import pickle
import random as rd
import time
import numpy as np
from scipy.special import expit
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

from nash_detect import *
from Utils.eval_helper import *
from Utils.yelpFeatureExtraction import *


"""
	Training Nash-Detect (Figure 3 in paper)
"""


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
	episodes = 50  # total episodes for training
	run_time = 'time1'

	print('dataset {} runtime is {}'.format(dataset_name, run_time))

	attacks = ['IncBP', 'IncDS', 'IncPR', 'Random', 'Singleton']
	detectors = ['GANG', 'Prior', 'SpEagle', 'fBox', 'Fraudar']

	setting1 = 'Training/' + dataset_name + '/IncBP.pickle'
	setting2 = 'Training/' + dataset_name + '/IncDS.pickle'
	setting3 = 'Training/' + dataset_name + '/Random.pickle'
	setting4 = 'Training/' + dataset_name + '/IncPR.pickle'
	setting5 = 'Training/' + dataset_name + '/Singleton.pickle'

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

	# uniformly initialize P and Q
	attacks_p = {}
	detectors_q = {}
	for a in attacks:
		attacks_p[a] = 1 / len(attacks)
	for d in detectors:
		detectors_q[d] = 1 / len(detectors)

	# initialize loggers
	posted_reviews = {}
	item_attack_mapping = {}
	remain_reviews_log = {}
	attack_log, detector_log = {a: [] for a in attacks_p.keys()}, {d: [] for d in detectors_q.keys()}
	loss_log, pe_log, rewards_log = [], [], []

	init_time = time.time()

	# start attacks and defenses
	for i in range(episodes):
		print('Starting episode {}'.format(i))
		posted_reviews[i] = []
		item_attack_mapping[i] = {}

		# generate fake reviews for each item
		for item in targets:
			attack = e_greedy_sample(attacks_p, epsilon)
			item_attack_mapping[i][item] = attack
			reviews = item_attack_review[item][attack]
			posted_reviews[i] += reviews
			# print(attack)

		# run all detectors on all reviews and added reviews
		r_spam_beliefs, r_accu_spam_beliefs, r_detector_belief, top_k_reviews, new_prod_user_graph, added_reviews = run_detectors(user_product_graph, prod_user_graph, posted_reviews[i], detectors_q, top_k)

		# remove the top_k reviews and calculate the practical effect
		print('Compute original Revenue ...')
		ori_RI, ori_ERI, ori_Revenue = compute_re(prod_user_graph, targets, elite_accounts)
		remain_product_user_graph = remove_topk_reviews(new_prod_user_graph, top_k_reviews)
		print('Compute new Revenue ...')
		new_RI, new_ERI, new_Revenue = compute_re(remain_product_user_graph, targets, elite_accounts)

		# calculate the cost of each posted review
		print('Compute cost ...')
		cost, remain_reviews = compute_cost([ori_RI, ori_ERI, ori_Revenue], [new_RI, new_ERI, new_Revenue], added_reviews, top_k_reviews, targets, elite_accounts)

		# compute reward for each item method
		print('Compute reward ...')
		reward = compute_reward([ori_RI, ori_ERI, ori_Revenue], [new_RI, new_ERI, new_Revenue], targets)

		# update sampling probabilities of attacks
		print('Update attack P ...')
		attacks_p = update_p(attacks_p, item_attack_mapping[i], reward, lr2)
		attacks_p = normalize(attacks_p)

		# update weights of detectors
		print('Update defense Q ...')
		detectors_q = update_q(detectors_q, cost, r_detector_belief, r_accu_spam_beliefs, remain_reviews, lr1)
		# detectors_q = normalize(detectors_q)

		# calculate total loss and total practical metric
		total_pe, total_loss = compute_total(ori_Revenue, new_Revenue, targets, cost, r_spam_beliefs)

		print(attacks_p)
		print(detectors_q)
		print('Total PE is {}'.format(total_pe))
		print('Total loss is {}'.format(total_loss))

		# logging
		remain_reviews_log[i] = remain_reviews  # Figure 3(c)
		for a, p in attacks_p.items():
			attack_log[a].append(p)  # Figure 3(a)
		for d, q in detectors_q.items():
			detector_log[d].append(q)  # Figure 3(b)
		loss_log.append(total_loss)  # Figure 3(f)
		pe_log.append(total_pe)  # Figure 3(d)
		rewards_log.append(sum(reward.values()))  # Figure 3(e)

		new_time = time.time()
		print('\nTime cost for episode {} is {}.\n'.format(i, new_time-init_time))
		init_time = new_time

	pickle.dump([attack_log, detector_log, loss_log, pe_log, remain_reviews_log, rewards_log], open('Training/' + dataset_name + '/' + run_time + '_' + str(episodes) + '_' + 'nash_detect.pickle', 'wb'))

