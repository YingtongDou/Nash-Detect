import copy
import itertools
import time

from gang import *
from iohelper import *
from yelpFeatureExtraction import *


def bp_evasion(user_product_graph, product_user_graph, priors, controlled_accounts, num_reviews, targets):
	"""

	:param user_product_graph:
	:param product_user_graph:
	:param priors: priors[0] -  user_priors; priors[2] - prod_priors; priors[1] - review_priors
	:param controlled_accounts: a set of controlled elite accounts
	:param num_reviews: number of reviews needed for each target
	:param targets: a set of target products
	:return:
	"""
	count = 0 # new edge counter
	added_edges = []
	account_log = []
	unique = 0 # selected unique accounts counter
	mean_priors = [0.5, 0.5, 0.5]
	t0 = time.time()

	# feature and prior calculation
	feature_suspicious_filename = 'feature_configuration.txt'
	review_feature_list = ['RD', 'EXT', 'EXT', 'DEV', 'ETF', 'ISR']
	user_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'BST', 'ERD', 'ETG']
	product_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'ERD', 'ETG']
	feature_config = load_feature_config(feature_suspicious_filename)
	feature_extractor = FeatureExtractor()
	UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.construct_all_features(user_product_graph, product_user_graph)
	user_ground_truth, review_ground_truth = create_ground_truth(user_product_graph)

	# initiialize the GANG model
	global_gang = gang(user_product_graph, product_user_graph, user_ground_truth, review_ground_truth,
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
			for i, p in priors[2].items():
				priors[2][i] = (p - p_min) / (p_max - p_min)
			for i, p in priors[1].items():
				priors[1][i] = (p - r_min) / (r_max - r_min)

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

		# the list saving the copies of the global GANGs for testing cost of adding each edge
		test_gangs = list(itertools.repeat(copy.deepcopy(global_gang), len(controlled_accounts)))

		# time counter
		t1 = time.time()
		print(str(t1 - t0))

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
			print('Total number of selected unique accounts: %d' % (new))
			account_log.append(added_account)

			# add the added edges to temporal new graph
			new_user_graph[added_account] = []
			new_user_graph[added_account].append((target, 1, -1, '2012-06-01'))
			new_prod_graph[target].append((added_account, 1, -1, '2012-06-01'))

			count += 1
			print('%d edges added.' % (count))


	return added_edges, user_product_graph, priors
