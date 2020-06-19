import time
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

from Detector.gang import *
from Utils.eval_helper import *
from Utils.yelpFeatureExtraction import *


"""
	The implementation of the IncBP attack.
"""


def bp_evasion(user_product_graph, product_user_graph, priors, controlled_accounts, num_reviews, targets, feature_config):
	"""
	:param user_product_graph: key = user_id, value = list of review tuples
	:param product_user_graph: key = product_id, value = list of review tuples
	:param priors: priors[0] -  user_priors; priors[1] - review_priors; priors[2] - prod_priors
	:param controlled_accounts: a set of controlled elite accounts
	:param num_reviews: number of reviews needed for each target
	:param targets: a set of target products
	"""
	count = 0 # new edge counter
	added_edges = []
	account_log = []
	unique = 0 # selected unique accounts counter
	t0 = time.time()

	# feature and prior calculation
	review_feature_list = ['RD', 'EXT', 'EXT', 'DEV', 'ETF', 'ISR']
	user_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'BST', 'ERD', 'ETG']
	product_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'ERD', 'ETG']
	feature_extractor = FeatureExtractor()
	UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.construct_all_features(user_product_graph, product_user_graph)
	user_ground_truth, review_ground_truth = create_ground_truth(user_product_graph)

	# normalize the priors for GANG
	priors, mean_priors = nor_priors(priors)

	# initiialize the GANG model
	global_gang = GANG(user_product_graph, product_user_graph, user_ground_truth,
					   priors, mean_priors, 0.1, nor_flg=True, sup_flg=False)
	
	# run Linearized Belief Propagation with GANG
	global_gang.pu_lbp(1000)

	# get node posterior
	global_posterior = global_gang.res_pu_spam_post_vector

	# go through each target to add new reviews (edges)
	for iter, target in enumerate(targets):
		# run or re-run GANG on the latest review graph (user_product_graph)
		if iter != 0:
			# calculate new features based original graph and temporal graph with new added edges 
			UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.update_all_features(user_product_graph, new_user_graph, product_user_graph, new_prod_graph, UserFeatures, ProdFeatures, ReviewFeatures)
			new_upriors = feature_extractor.calculateNodePriors(user_feature_list, UserFeatures, feature_config)
			new_ppriors = feature_extractor.calculateNodePriors(product_feature_list, ProdFeatures, feature_config)
			new_rpriors = feature_extractor.calculateNodePriors(review_feature_list, ReviewFeatures, feature_config)
			priors = [new_upriors, new_rpriors, new_ppriors]

			# normalize the node priors to (0,1)
			# if we normalize the prior, we need to set nor_flg to True for the gang model
			priors, mean_priors = nor_priors(priors)

			# update the global graph and global GANG model
			for key, items in new_prod_graph.items():
				global_gang.res_pu_spam_prior_vector[int(key)] = priors[2][key]
				global_gang.diag_pu_matrix[int(key), int(key)] += len(new_prod_graph[key])
				for item in items:
					product_user_graph[key].append((item[0], 1, 1, '2012-06-01'))
					user_product_graph[item[0]].append((key, 1, 1, '2012-06-01'))
					global_gang.pu_matrix[int(key), int(item[0])] = 1
					global_gang.pu_matrix[int(item[0]), int(key)] = 1
					global_gang.diag_pu_matrix[int(item[0]), int(item[0])] += 1
					global_gang.res_pu_spam_prior_vector[int(item[0])] = priors[0][item[0]]

			# update the global model node posteriors
			global_gang.pu_lbp(1)
			print('---')
			print(sum([global_posterior[int(i)] - global_gang.res_pu_spam_post_vector[int(i)] for i in controlled_accounts]))
			print('---')
			global_posterior = global_gang.res_pu_spam_post_vector


		# initialize the temporal graphs that carry the new added edges
		new_user_graph = {}
		new_prod_graph = {}
		new_prod_graph[target] = []

		# time counter
		t1 = time.time()
		print('attacking time for target {} is {}'.format(iter, str(t1 - t0)))

		# select the accounts with minimum posteriors estimated by GANG
		selected_accounts = [(account, global_posterior[int(account)]) for account in controlled_accounts]

		selected_accounts = sorted(selected_accounts, reverse=False, key=lambda x: x[1])

		selected_accounts = [account[0] for account in selected_accounts[:num_reviews]]

		for added_account in selected_accounts:
			added_edges.append((added_account, target))
			print((added_account, target))

			# count no. of new selected accounts
			if added_account not in account_log:
				unique += 1
			print('Total number of selected unique accounts: %d' % (unique))
			account_log.append(added_account)

			# add the added edges to temporal new graph
			new_user_graph[added_account] = []
			new_user_graph[added_account].append((target, 1, -1, '2012-06-01'))
			new_prod_graph[target].append((added_account, 1, -1, '2012-06-01'))

			count += 1
			print('%d edges added.' % (count))


	return added_edges, user_product_graph, priors
