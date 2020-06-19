import copy as cp
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

from Utils.yelpFeatureExtraction import *
from Utils.iohelper import *


"""
	Define several functions for evaluation.
"""


def create_ground_truth(user_data):
	"""Given user data, return a dictionary of labels of users and reviews
	Args:
		user_data: key = user_id, value = list of review tuples.

	Return:
		user_ground_truth: key = user id (not prefixed), value = 0 (non-spam) /1 (spam)
		review_ground_truth: review id (not prefixed), value = 0 (non-spam) /1 (spam)
	"""
	user_ground_truth = {}
	review_ground_truth = {}

	for user_id, reviews in user_data.items():

		user_ground_truth[user_id] = 0

		for r in reviews:
			prod_id = r[0]
			label = r[2]

			if label == -1:
				review_ground_truth[(user_id, prod_id)] = 1
				user_ground_truth[user_id] = 1
			else:
				review_ground_truth[(user_id, prod_id)] = 0

	return user_ground_truth, review_ground_truth


def create_evasion_ground_truth(user_data, evasive_spams):
	"""Assign label 1 to evasive spams and 0 to all existing reviews; Assign labels to accounts accordingly
	Args:
		user_data: key = user_id, value = list of review tuples.
			user_data can contain only a subset of reviews
			(for example, if some of the reviews are used for training)

		evasive_spams: key = product_id, value = list of review tuples

	Return:
		user_ground_truth: key = user id (not prefixed), value = 0 (non-spam) /1 (spam)
		review_ground_truth: review id (not prefixed), value = 0 (non-spam) /1 (spam)
	"""
	old_spammers = set()
	old_spams = set()

	user_ground_truth = {}
	review_ground_truth = {}

	# assign label 0 to all existing reviews and users
	for user_id, reviews in user_data.items():
		user_ground_truth[user_id] = 0

		for r in reviews:
			prod_id = r[0]
			label = r[2]
			review_ground_truth[(user_id, prod_id)] = 0

			if label == -1:
				old_spams.add((user_id, prod_id))
				old_spammers.add(user_id)

	# exclude previous spams and spammers, since the controlled accounts are selcted from the normal accounts.
	for r_id in old_spams:
		review_ground_truth.pop(r_id)
	for u_id in old_spammers:
		user_ground_truth.pop(u_id)

	# add label 1 to the evasive spams
	for prod_id, spams in evasive_spams.items():

		for r in spams:
			user_id = r[0]

			review_ground_truth[(user_id, prod_id)] = 1
			# this user now has posted at least one spam, so set its label to 1
			user_ground_truth[user_id] = 1

	return user_ground_truth, review_ground_truth


def evaluate(y, pred_y):
	"""
	Evaluate the prediction of account and review
	Args:
		y: dictionary with key = user_id/review_id and value = ground truth (1 means spam, 0 means non-spam)

		pred_y: dictionary with key = user_id/review_id and value = p(y=spam | x) produced by SpEagle.
				the keys in pred_y must be a subset of the keys in y
	"""
	posteriors = []
	ground_truth = []

	for k, v in pred_y.items():
		if k in y:
			posteriors.append(v)
			ground_truth.append(y[k])

	auc = roc_auc_score(ground_truth, posteriors)
	ap = average_precision_score(ground_truth, posteriors)

	return auc, ap


def scale_value(value_dict):
	"""
	Calculate and return a dict of the value of input dict scaled to (0, 1)
	"""

	ranked_dict = [(user, value_dict[user]) for user in value_dict.keys()]
	ranked_dict = sorted(ranked_dict, reverse=True, key=lambda x: x[1])

	up_max, up_mean, up_min = ranked_dict[0][1], ranked_dict[int(len(ranked_dict) / 2)][1], ranked_dict[-1][1]

	scale_dict = {}
	for i, p in value_dict.items():
		norm_value = (p - up_min) / (up_max - up_min)

		if norm_value == 0: # avoid the 0
			scale_dict[i] = 0 + 1e-7
		elif norm_value == 1: # avoid the 1
			scale_dict[i] = 1 - 1e-7
		else:
			scale_dict[i] = norm_value

	return scale_dict


def normalize(dict):
	"""
	Normalized given dictionary value to [0,1] and sum(dict.values()) = 1
	"""

	total = sum([v for v in dict.values()])

	for i, v in dict.items():
		dict[i] = v/total

	return dict


def add_adversarial_review(user_product_graph, prod_user_graph, new_edges):
	"""
	Add fake reviews for adversarial training
	"""

	# Promotion or demotion setting
	rating = 5

	new_user_graph = {}
	new_product_graph = {}
	single_mapping = {}

	feature_suspicious_filename = 'Utils/feature_configuration.txt'
	review_feature_list = ['RD', 'EXT', 'DEV', 'ETF', 'ISR']
	user_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'BST', 'ERD', 'ETG']
	product_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'ERD', 'ETG']

	# read the graph and node priors
	feature_config = load_feature_config(feature_suspicious_filename)
	feature_extractor = FeatureExtractor()

	# load new edges to a new graph to accelerate the prior computation
	for added_edge in new_edges:
		added_account = added_edge[0]
		target = added_edge[1]
		# add elite reviews
		if added_account in user_product_graph.keys():
			origin_date = user_product_graph[added_account][0][3]
			new_date = origin_date
			new_rating = rating
			if added_account not in new_user_graph.keys():
				# a tuple of (product_id, rating, label, posting_date)
				new_user_graph[added_account] = [(target, new_rating, -1, new_date)]
			else:
				new_user_graph[added_account].append((target, new_rating, -1, new_date))
			if target not in new_product_graph.keys():
				# a tuple of (user_id, rating, label, posting_date)
				new_product_graph[target] = [(added_account, new_rating, -1, new_date)]
			else:
				new_product_graph[target].append((added_account, new_rating, -1, new_date))
		else:
			# add singleton reviews
			origin_date = '2012-06-01'
			new_date = origin_date
			new_rating = rating
			new_account = str(len(user_product_graph) + len(prod_user_graph))
			single_mapping[added_account] = new_account
			user_product_graph[new_account] = [(target, new_rating, -1, new_date)]
			prod_user_graph[target].append((new_account, new_rating, -1, new_date))


	# calculate feature_extractorres on the complete graph
	UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.construct_all_features(user_product_graph,
																						  prod_user_graph)


	if len(new_user_graph) != 0:
		# update features with the new graph
		UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.update_all_features(user_product_graph,
																						   new_user_graph,
																						   prod_user_graph,
																						   new_product_graph,
																						   UserFeatures,
																						   ProdFeatures, ReviewFeatures)

	new_upriors = feature_extractor.calculateNodePriors(user_feature_list, UserFeatures, feature_config)
	new_ppriors = feature_extractor.calculateNodePriors(product_feature_list, ProdFeatures, feature_config)
	new_rpriors = feature_extractor.calculateNodePriors(review_feature_list, ReviewFeatures, feature_config)

	user_priors = new_upriors
	review_priors = cp.deepcopy(new_rpriors)
	prod_priors = new_ppriors

	# create a set of new added reviews
	evasive_spams = {}
	new_added_reviews = []
	for added_edge in new_edges:
		added_account = added_edge[0]
		target = added_edge[1]
		if added_account in single_mapping.keys():
			added_account = single_mapping[added_account]
		if target not in evasive_spams.keys():
			evasive_spams[target] = [(added_account, rating, -1, '2012-06-01')]
		else:
			evasive_spams[target].append((added_account, rating, -1, '2012-06-01'))

		new_added_reviews.append((added_account, target))

	# add new edges into the original graph
	for e in new_edges:
		u_id = str(e[0])
		p_id = str(e[1])
		if u_id not in single_mapping.keys():
			user_product_graph[u_id].append((p_id, rating, -1, '2012-06-01'))
			prod_user_graph[p_id].append((u_id, rating, -1, '2012-06-01'))

	# create evasion ground truth
	user_ground_truth, review_ground_truth = create_evasion_ground_truth(user_product_graph, evasive_spams)

	return [user_priors, review_priors, prod_priors], user_product_graph, prod_user_graph, user_ground_truth, review_ground_truth, new_added_reviews


def nor_priors(priors):
	"""
	Normalize the node priors for GANG
	:return:
	"""
	new_upriors, new_rpriors, new_ppriors = priors

	# normalize the node priors to (0,1)
	# if we normalize the prior, we need to set nor_flg to True for the gang model
	ranked_upriors = [(user, new_upriors[user]) for user in new_upriors.keys()]
	ranked_upriors = sorted(ranked_upriors, reverse=True, key=lambda x: x[1])
	ranked_rpriors = [(user, new_rpriors[user]) for user in new_rpriors.keys()]
	ranked_rpriors = sorted(ranked_rpriors, reverse=True, key=lambda x: x[1])
	ranked_ppriors = [(user, new_ppriors[user]) for user in new_ppriors.keys()]
	ranked_ppriors = sorted(ranked_ppriors, reverse=True, key=lambda x: x[1])
	u_max, u_mean, u_min = ranked_upriors[0][1], ranked_upriors[int(len(ranked_upriors) / 2)][1], ranked_upriors[-1][1]
	p_max, p_mean, p_min = ranked_ppriors[0][1], ranked_ppriors[int(len(ranked_ppriors) / 2)][1], ranked_ppriors[-1][1]
	r_max, r_mean, r_min = ranked_rpriors[0][1], ranked_rpriors[int(len(ranked_rpriors) / 2)][1], ranked_rpriors[-1][1]
	for i, p in priors[0].items():
		priors[0][i] = (p - u_min) / (u_max - u_min)
	for i, p in priors[1].items():
		priors[1][i] = (p - r_min) / (r_max - r_min)
	for i, p in priors[2].items():
		priors[2][i] = (p - p_min) / (p_max - p_min)

	return priors, [u_mean, r_mean, p_mean]


def select_elite(user_prod_graph, threshold=10):
	"""
	Select a fixed set of elite accounts
	"""
	elite_accounts = []
	for user in user_prod_graph.keys():
		if len(user_prod_graph[user]) >= threshold:
			elite_accounts.append(user)

	return elite_accounts
