import copy as cp
import pickle
import random as rd
import time
import numpy as np
from scipy.special import expit
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

from Detector.eval_Fraudar import *
from Detector.eval_GANG import *
from Detector.eval_SpEagle import *
from Detector.eval_fBox import *
from Utils.iohelper import *
from Utils.eval_helper import *
from Utils.yelpFeatureExtraction import *


"""
	Nash-Detect implementation
"""


def e_greedy_sample(attacks_p, e=.2):
	"""
	e-greedy algorithm for sampling spamming attack strategies from discrete distributions
	"""

	z = np.random.random()

	attacks_p.keys()

	if z > e:  # sample attack based on their probabilities
		attack = np.random.choice(list(attacks_p), 1, p=list(attacks_p.values()))
		attack = attack[0]
	else:  # random sample
		attack = rd.choice(list(attacks_p))

	return attack


def load_fake_reviews(attacks, targets, paths):
	"""
	load fake reviews for each spamming attack to each item
	"""

	attack_item_review = {}

	for item in targets:
		attack_reviews = {}
		for attack in attacks:
			attack_reviews[attack] = []
			with open(paths[attack], 'rb') as f:
				evasions = pickle.load(f)
			for index, review in enumerate(evasions[2]):
				if review[1] == item:
					attack_reviews[attack] += evasions[2][index:index+15]
					break

		attack_item_review[item] = attack_reviews

	return attack_item_review


def run_detectors(upg, pug, added_reviews, detectors_q, top_k_per):
	"""
	Run detectors on all reviews plus added fake reviews.
	Return the aggregated review posterior beliefs and top_k suspicious reviews
	"""

	spammer_ids = []
	for review in added_reviews:
		if review[0] not in spammer_ids:
			spammer_ids.append(review[0])

	u_p_graph, p_u_graph = cp.deepcopy(upg), cp.deepcopy(pug)

	new_priors, u_p_graph, p_u_graph, user_ground_truth, review_ground_truth, new_added_reviews = add_adversarial_review(u_p_graph, p_u_graph, added_reviews)

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
	speagle_rbelief, fraudar_rbelief, fbox_rbelief, prior_rbelief = scale_value(speagle_rbelief), scale_value(fraudar_rbelief), scale_value(fbox_rbelief), scale_value(prior_rbelief)

	# weight of each detector
	gang_q, speagle_q, fraudar_q, fbox_q, prior_q = detectors_q['GANG'], detectors_q['SpEagle'], detectors_q['Fraudar'], detectors_q['fBox'], detectors_q['Prior']

	print('Compute Posterior ...')
	# compute aggregated review posteriors
	r_accu_spam_beliefs = {}
	r_spam_beliefs = {}
	r_detector_belief = {}

	for r in gang_rbelief.keys():
		accu_spam_belief = 0.0
		accu_spam_belief += gang_q * gang_rbelief[r]
		accu_spam_belief += speagle_q * speagle_rbelief[r]
		accu_spam_belief += fraudar_q * fraudar_rbelief[r]
		accu_spam_belief += fbox_q * fbox_rbelief[r]
		accu_spam_belief += prior_q * prior_rbelief[r]

		r_accu_spam_beliefs[r] = accu_spam_belief
		r_spam_beliefs[r] = expit(accu_spam_belief)  # 1 / (1 + math.exp(-accu_belief))
		r_detector_belief[r] = {'GANG': gang_rbelief[r], 'SpEagle': speagle_rbelief[r], 'Fraudar': fraudar_rbelief[r], 'fBox': fbox_rbelief[r], 'Prior': prior_rbelief[r]}

	# rank the top_k suspicious reviews
	ranked_rbeliefs = [(review, r_spam_beliefs[review]) for review in r_spam_beliefs.keys()]
	ranked_rbeliefs = sorted(ranked_rbeliefs, reverse=True, key=lambda x: x[1])
	top_k = int(len(r_spam_beliefs) * top_k_per)
	top_k_reviews = [review[0] for review in ranked_rbeliefs[:top_k]]
	print(len(top_k_reviews))
	# remove false positives
	for review in top_k_reviews:
		if review not in added_reviews:
			top_k_reviews.remove(review)

	print('top k is {}'.format(len(top_k_reviews)))

	return r_spam_beliefs, r_accu_spam_beliefs, r_detector_belief, top_k_reviews, p_u_graph, new_added_reviews


def compute_re(product_user_graph, targets, elite_accounts):
	"""
	Compute the revenue estimation based on given reviews
	"""

	# calculate the original revenue
	avg_ratings = {}
	for product, reviews in product_user_graph.items():
		rating = 0
		if len(reviews) == 0:
			avg_ratings[product] = 0
		else:
			for review in reviews:
				rating += review[1]
			avg_ratings[product] = rating/len(reviews)

	mean_rating = sum(r for r in avg_ratings.values())/len(avg_ratings)
	RI = {}
	ERI = {}
	Revenue = {}

	for target in targets:
		RI[target] = avg_ratings[target] - mean_rating
		temp_ERI = []
		for review in product_user_graph[target]:
			if review[0] in elite_accounts:
				temp_ERI.append(review[1])
		ERI[target] = sum(temp_ERI)/len(temp_ERI) if len(temp_ERI) != 0 else 0

	# Eq. (1) in paper
	for target in targets:
		Revenue[target] = 0.09 + 0.035 * RI[target] + 0.036 * ERI[target]

	return RI, ERI, Revenue


def remove_topk_reviews(product_user_graph, filter_reviews):
	"""
	Remove top_k suspicious reviews from the given review graph
	"""

	for edge in filter_reviews:
		for review in product_user_graph[edge[1]]:
			if review[0] == edge[0]:
				product_user_graph[edge[1]].remove(review)

	return product_user_graph


def compute_cost(ori, new, added_reviews, filter_reviews, target_ids, elite_accounts):
	"""
	Compute the cost of detectors according to the difference of the practical metric after and before the attacks&defenses
	"""

	ori_RI, ori_ERI, ori_Revenue = ori
	new_RI, new_ERI, new_Revenue = new

	remain_reviews = [r for r in added_reviews if r not in filter_reviews]

	elite_reviews = []
	item_review_count = {}
	item_elite_count = {}

	for item in target_ids:
		item_review_count[item] = 0
		item_elite_count[item] = 0

	for review in remain_reviews:
		item = review[1]
		if review[0] in elite_accounts:
			elite_reviews.append(review)
			item_review_count[item] += 1
			item_elite_count[item] += 1
		else:
			item_review_count[item] += 1

	cost = {}

	# Eq. (4) in paper
	for review in remain_reviews:
		item = review[1]
		if new_Revenue[item] - ori_Revenue[item] <= 0:
			cost[review] = 0
		elif review in elite_reviews:
			cost[review] = 0.035*(new_RI[item]-ori_RI[item])/item_review_count[item]\
						   + 0.036*(new_ERI[item]-ori_ERI[item])/item_elite_count[item]
		else:
			cost[review] = 0.035*(new_RI[item]-ori_RI[item])/item_review_count[item]

	return cost, remain_reviews


def compute_reward(ori, new, target_ids):
	"""
	Compute the reward for each target item
	"""

	reward = {}
	PE_dict = {}
	ori_RI, ori_ERI, ori_Revenue = ori
	new_RI, new_ERI, new_Revenue = new
	max_PE, min_PE, total_PE = 0, 0, 0

	for item in target_ids:
		PE = new_Revenue[item] - ori_Revenue[item]
		# Eq. (3) in paper
		PE = min_PE if PE <= min_PE else PE
		max_PE = PE if PE >= max_PE else max_PE
		total_PE += PE
		PE_dict[item] = PE

	avg_PE = total_PE/len(target_ids)
	PE_interval = max_PE - min_PE

	# Eq. (9) in paper
	for item, PE in PE_dict.items():
		reward[item] = expit((PE - avg_PE)/PE_interval)
	return reward


def update_p(attacks_p, item_attack_mapping, reward, lr2):
	"""
	Updating the sampling probabilities of attacks according to the rewards computed before
	:param lr2: the learning rate of attacks_p (\eta in Eq. (10))
	"""

	# record the selection times and reward of attack during each update
	counter = {a: 0 for a in attacks_p.keys()}
	reward_log = cp.deepcopy(counter)

	# update the P with average reward. Eq. (10) in the paper
	for i, current_reward in reward.items():
		current_attack = item_attack_mapping[i]
		total_count, total_reward = counter[current_attack], reward_log[current_attack]

		# compute the average accumulated reward recursively
		new_reward = total_reward + 1/(total_count+1) * (current_reward - total_reward)

		# update the P with new reward
		attacks_p[current_attack] += lr2 * new_reward

		# update counter and reward log
		counter[current_attack] += 1
		reward_log[current_attack] = new_reward

	for a, p in attacks_p.items():
		if p <= 0:
			attacks_p[a] = 0

	return attacks_p


def update_q(detectors_q, cost, r_detector_belief, r_accu_spam_beliefs, remain_reviews, lr1):
	"""
	Updating the importance of detectors based on their gradients of the cost-sensitive loss function
	"""

	# gradient decent updating according to of Eq. (11) in paper
	for d, q in detectors_q.items():
		grad_sum = 0
		for review in remain_reviews:
			# calculate gradient for current q (detector weight)
			grad_sum += -1 * cost[review] * r_detector_belief[review][d] * expit(-r_accu_spam_beliefs[review])

		grad_norm = grad_sum / len(remain_reviews)
		# update detector importance
		detectors_q[d] = q - lr1 * grad_norm

	return detectors_q


def compute_total(ori_Revenue, new_Revenue, targets, cost, r_spam_beliefs):
	"""
	Compute total loss of detectors and total practical effort of attacks
	"""

	total_pe = sum([new_Revenue[i]-ori_Revenue[i] for i in targets])

	total_loss = sum([-1 * cost[r] * np.log(r_spam_beliefs[r]) for r in cost.keys()])/len(cost)

	return total_pe, total_loss
