import time
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

from Utils.yelpFeatureExtraction import *


"""
	The implementation of the IncPR attack.
"""


def pr_evasion(user_product_graph, product_user_graph, c, r, t, feature_config):
	"""
	Args:
		user_product_graph: key = user_id, value = list of review tuples
		product_user_graph: key = product_id, value = list of review tuples
		c: list of controlled accounts
		r: number of reviews to be posted each account
		t: target list
		feature_config: feature configuration file
	"""

	# total number of spams posted 
	count = 0
	added_edges = []
	t0 = time.time()

	# total number of selected unique accounts
	unique = 0
	# feature name list for account node
	user_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'BST', 'ERD', 'ETG']

	new_user_graph = {}
	new_product_graph = {}

	# compute the features and account priors
	feature_extractor = FeatureExtractor()
	UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.construct_all_features(user_product_graph, product_user_graph)
	new_upriors = feature_extractor.calculateNodePriors(user_feature_list, UserFeatures, feature_config)

	account_log = []

	# for each target, find controlled accounts to post spams
	for target in t:

		# select r accounts with minimum priors
		selected_accounts = [(account, new_upriors[account]) for account in c]
		selected_accounts = sorted(selected_accounts, reverse = False, key=lambda x:x[1])
		selected_accounts = [account[0] for account in selected_accounts[:r]]

		# count the number of selected unique accounts
		for account in selected_accounts:
			if account not in account_log:
				unique += 1
		print('Total number of selected unique accounts: %d' %(unique))

		account_log = account_log + selected_accounts

		# add the added_edges to the global graph
		for added_account in selected_accounts:
			if added_account not in new_user_graph.keys():
				# a tuple of (product_id, rating, label, posting_date)
				new_user_graph[added_account] = [(target, 1, -1, '2012-06-01')]
			else:
				new_user_graph[added_account].append((target, 1, -1, '2012-06-01'))
			if target not in new_product_graph.keys():
				# a tuple of (user_id, rating, label, posting_date)
				new_product_graph[target] = [(added_account, 1, -1, '2012-06-01')]
			else:
				new_product_graph[target].append((added_account, 1, -1, '2012-06-01'))

		# calculate new features and priors with added edges
		UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.update_all_features(user_product_graph, new_user_graph, product_user_graph, new_product_graph, UserFeatures, ProdFeatures, ReviewFeatures)
		new_upriors = feature_extractor.calculateNodePriors(user_feature_list, UserFeatures, feature_config)

		# save the added edges
		for added_account in selected_accounts:
			review_id = (added_account, target)
			added_edges.append(review_id)
		
		t1 = time.time()
		
		print('Time consumed: {}'.format(str(t1-t0)))
		print('\n---------------------------------\n')

	return added_edges, user_product_graph