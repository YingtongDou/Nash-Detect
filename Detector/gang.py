"""
	Implement GANG on a User-Review-Product graph.
"""

import random
import time

import matplotlib.pyplot as plt
import numpy as np
from eval_helper import *
from iohelper import *
from scipy.sparse import lil_matrix
from yelpFeatureExtraction import *


class local_gang():
	def __init__(self, global_posterior, user_product_graph, product_user_graph, priors, max_iters=100):

		# max # of iterations
		self.max_iters = max_iters
		# number of dimensions of product-user matrix
		self.pu_dim = len(priors[0])+len(priors[2])
		# spam belief prior vector
		self.res_pu_spam_prior_vector = None
		# product-user posterior belief vector
		self.res_pu_post_vector = np.zeros((self.pu_dim, 1))
		# the pu adjacency matrix
		self.pu_matrix = None
		# node_id-matrix_id mapping dictionary
		self.id_mapping = {}
		# local graph user_product_graph
		self.user_graph = user_product_graph
		self.prod_graph = product_user_graph
		# sparse row matrix is faster when multiply with vectors
		self.pu_csr_matrix = None
		self.diag_pu_csr_matrix = None
		self.nor_pu_csr_matrix = None
		# priors dictionary
		self.r_priors = priors[1]
		self.u_priors = priors[0]
		self.p_priors = priors[2]
		# build prior belief vector
		p_vector, u_vector = [], []

		# the mean value with normalization
		u_mean, p_mean = 0.5, 0.5

		# initialize prior vector and id mapping
		i = 0
		for pid, p_prior in priors[2].items():
			self.id_mapping[pid] = i
			p_vector.append(p_prior)
			i += 1
		j = 0
		for uid, u_prior in priors[0].items():
			self.id_mapping[uid] = j+len(priors[1])
			u_vector.append(u_prior)
			j += 1

		res_u_vector = [i-u_mean for i in u_vector]
		res_p_vector = [i-p_mean for i in p_vector]

		# aggregate the prior vectors
		res_pu_vector = res_p_vector + res_u_vector
		# res_pur_vector = res_pu_vector + res_r_vector

		self.res_pu_spam_prior_vector = np.c_[res_pu_vector]
		# self.res_pur_prior_vector = np.array(res_pur_vector)

		# build product-user adjacency sparse matrix
		self.pu_matrix = lil_matrix((self.pu_dim, self.pu_dim))
		self.diag_pu_matrix = lil_matrix((self.pu_dim, self.pu_dim))

		for p_id, reviews in self.prod_graph.items():
			# copy the global posterior information
			self.res_pu_post_vector[self.id_mapping[p_id], 0] = global_posterior[int(p_id), 0]
			# build diagonal matrix
			self.diag_pu_matrix[self.id_mapping[p_id], self.id_mapping[p_id]] = len(self.prod_graph[p_id])
			for r in reviews:
				self.pu_matrix[self.id_mapping[p_id], self.id_mapping[r[0]]] = 1

		for u_id, reviews in self.user_graph.items():
			# copy the global posterior information
			self.res_pu_post_vector[self.id_mapping[u_id], 0] = global_posterior[int(u_id), 0]
			# build diagonal matrix
			self.diag_pu_matrix[self.id_mapping[u_id], self.id_mapping[u_id]] = len(self.user_graph[u_id])
			for r in reviews:
				self.pu_matrix[self.id_mapping[u_id], self.id_mapping[r[0]]] = 1

	def pu_lbp(self):
		'''
		Run the matrix form of lbp on the product-user sparse matrix
		:return: the posterior difference of the last round of lbp
		'''

		# transfer to sparse row matrix to accelerate calculation
		self.pu_csr_matrix = self.pu_matrix.tocsr()
		self.diag_pu_csr_matrix = self.diag_pu_matrix.tocsr()

		# normalization approach III: D^(-1/2)*A*D^(-1/2)
		D_12 = self.diag_pu_csr_matrix.power(-0.5)
		self.nor_pu_csr_matrix = D_12 * self.pu_csr_matrix * D_12

		i = 0
		while i < self.max_iters:
			sum_0 = np.sum(self.res_pu_post_vector)

			# print(self.pu_csr_matrix.get_shape())
			# print(self.res_pu_post_vector.__sizeof__())
			# print(self.res_pu_prior_vector.__sizeof__())
			self.res_pu_post_vector = self.res_pu_spam_prior_vector + 2 * 0.001 * (self.nor_pu_csr_matrix.dot(self.res_pu_post_vector))
			sum_1 = np.sum(self.res_pu_post_vector)

			# print('iter: ' + str(i))
			# print('diff: ' + str(abs(sum_0 - sum_1)))

			i += 1

			if abs(sum_0 - sum_1) < 0.1:
				return abs(sum_0 - sum_1)

	def classify(self):
		'''
		Calculate the posterior belief of three type of nodes
		:return: u_post: users posterior beliefs, p_post: products posterior beliefs,
		r_post: reviews posterior beliefs.
		'''
		u_post = {}
		p_post = {}
		r_post = {}
		pu_post = self.res_pu_post_vector
		no_prod = len(self.p_priors)
		# extract the posterior belief of users and reviews
		for i, r in enumerate(pu_post[no_prod:]):
			u_post[str(i + no_prod)] = r
		for i, r in enumerate(pu_post[:no_prod]):
			p_post[str(i)] = r
		for i, r in self.r_priors.items():
			r_post[i] = (u_post[str(self.id_mapping[i[0]])] + r) / 2

		return u_post, p_post, r_post

class gang():

	def __init__(self, user_product_graph, product_user_graph, user_ground_truth, review_ground_truth,
				 priors, mean_priors, sup_per, nor_flg, sup_flg=False):

		# number of dimensions of product-user matrix
		self.pu_dim = len(priors[0])+len(priors[2])
		# number of dimensions of product-user-review matrix
		# self.pur_dim = len(priors[0])+len(priors[1])+len(priors[2])
		# spam belief prior vector
		self.res_pu_spam_prior_vector = None
		# diagonal matrix used for normalization
		self.diag_pu_matrix = None
		# product-user spam posterior belief vector
		self.res_pu_spam_post_vector = np.zeros((self.pu_dim, 1))
		# # benign belief vector
		# self.res_pu_benign_prior_vector = None
		# # product-user benign posterior belief vector
		# self.res_pu_benign_post_vector = np.zeros((self.pu_dim,), dtype=int)
		# sparse row matrix is faster when multiply with vectors
		self.pu_csr_matrix = None
		self.diag_pu_csr_matrix = None
		self.nor_pu_csr_matrix = None
		# priors dictionary
		self.r_priors = priors[1]
		self.u_priors = priors[0]
		self.p_priors = priors[2]
		# # product-user-review posterior belief vector
		# self.res_pur_post_vector = np.zeros((self.pur_dim,), dtype=int)
		#
		# # review index in pur matrix
		# review_id = {}
		# for iter, r in enumerate(priors[2]):
		# 	review_id[r] = iter+self.pu_dim

		# build prior belief vector
		p_vector, u_vector, r_vector = [], [], []

		if nor_flg:
			# the mean value with normalization
			u_mean, p_mean, r_mean = 0.5, 0.5, 0.5
		else:
			# the mean value without normalization
			u_mean, p_mean, r_mean = mean_priors[0], mean_priors[2], mean_priors[1]

		for p in priors[2].values():
			p_vector.append(p)
		for u in priors[0].values():
			u_vector.append(u)
		# for r in priors[2].values():
		# 	r_vector.append(r)

		res_u_vector = [i-u_mean for i in u_vector]
		res_p_vector = [i-p_mean for i in p_vector]
		# res_r_vector = [i-r_mean for i in r_vector]

		# add semi-supervised review information
		# pos_ids, neg_ids = self.semi_data(review_ground_truth, 0.02)
		# for iter, prob in enumerate(res_r_vector):
		# 	if iter in pos_ids:
		# 		res_r_vector[iter] = 1 - r_mean
		# 	elif iter in neg_ids:
		# 		res_r_vector[iter] = 0 - r_mean

		# add semi-supervised user information
		if sup_flg:
			pos_ids, neg_ids = self.semi_data(user_ground_truth, sup_per)
			for iter, prob in enumerate(res_u_vector):
				if iter in pos_ids:
					res_u_vector[iter] = 1 - u_mean
				elif iter in neg_ids:
					res_u_vector[iter] = 0 - u_mean

		# aggregate the prior vectors
		res_pu_vector = res_p_vector + res_u_vector
		# res_pur_vector = res_pu_vector + res_r_vector

		self.res_pu_spam_prior_vector = np.c_[res_pu_vector]

		# self.res_pur_prior_vector = np.array(res_pur_vector)
		# self.res_pu_benign_prior_vector = 1 - self.res_pu_spam_prior_vector

		# build product-user adjacency sparse matrix
		self.pu_matrix = lil_matrix((self.pu_dim, self.pu_dim))
		# create the pu diagonal matrix
		self.diag_pu_matrix = lil_matrix((self.pu_dim, self.pu_dim))
		for id in range(0, self.pu_dim):
			if id < len(self.p_priors):
				self.diag_pu_matrix[id, id] = len(product_user_graph[str(id)])
			else:
				self.diag_pu_matrix[id, id] = len(user_product_graph[str(id)])

		for p_id, reviews in product_user_graph.items():

			for r in reviews:
				self.pu_matrix[int(p_id), int(r[0])] = 1#/len(product_user_graph[p_id])

		for u_id, reviews in user_product_graph.items():

			for r in reviews:
				self.pu_matrix[int(u_id), int(r[0])] = 1#/len(user_product_graph[u_id])

		# normalization approach I:  A*D^(-1)
		# D_1 = self.diag_pu_csr_matrix.power(-1)
		# self.pu_csr_matrix = self.pu_csr_matrix * D_1

		# normalization approach II: D^(-1)*A
		# D_1 = self.diag_pu_csr_matrix.power(-1)
		# self.pu_csr_matrix = D_1 * self.pu_csr_matrix

		# normalization approach III: D^(-1/2)*A*D^(-1/2)
		# D_12 = self.diag_pu_csr_matrix.power(-0.5)
		# self.pu_csr_matrix = D_12 * self.pu_csr_matrix * D_12

		# # build product-user-review adjacency sparse matrix
		# self.pur_matrix = lil_matrix((self.pur_dim, self.pur_dim))
		#
		# for p_id, reviews in product_user_graph.items():
		#
		# 	for r in reviews:
		# 		self.pur_matrix[int(p_id), review_id[(r[0], p_id)]] = 1
		#
		# for u_id, reviews in user_product_graph.items():
		#
		# 	for r in reviews:
		# 		self.pur_matrix[int(u_id), review_id[(u_id, r[0])]] = 1
		#
		# for r in priors[2].keys():
		#
		# 	self.pur_matrix[review_id[r], int(r[0])] = 1
		# 	self.pur_matrix[review_id[r], int(r[1])] = 1

	# def add_new_data(self, new_user_product_graph, new_priors):
	# 	'''
	#
	# 	:param new_user_product_graph:
	# 	:param new_priors:
	# 	:return:
	# 	'''
	#
	# 	for user_id, reviews in new_user_product_graph.items():
	# 		self.res_pu_prior_vector[user_id] = new_priors[0][user_id]
	# 		for review in reviews:
	# 			self.r_priors[(user_id, review[0])] = new_priors[2][(user_id, review[0])]
	# 			self.res_pu_prior_vector[review[0]] = new_priors[1][review[0]]
	# 			self.pu_matrix[user_id, review[0]] = 1
	# 			self.pu_matrix[review[0], user_id] = 1

	def semi_data(self, ground_truth, portion):
		'''
		produce the sampled labeled review id used for semi-supervised prior
		:param ground_truth: dict of ground truth {uid:label} or {rid:label}
		:param portion: portion of the labeled data
		:return: review id which are used for supervising
		'''

		smaple_size = int(len(ground_truth) * portion * 0.5)
		total_list = [r for r in ground_truth.keys()]
		pos_list = []
		neg_list = []
		for id, label in ground_truth.items():
			if label == 1:
				pos_list.append(id)
			else:
				neg_list.append(id)

		pos_sample = [pos_list[i] for i in sorted(random.sample(range(len(pos_list)), smaple_size))]
		neg_sample = [neg_list[i] for i in sorted(random.sample(range(len(neg_list)), smaple_size))]

		pos_ids = [total_list.index(s) for s in pos_sample]
		neg_ids = [total_list.index(s) for s in neg_sample]

		return pos_ids, neg_ids

	def pu_lbp(self, max_iters):
		'''
		Run the matrix form of lbp on the product-user sparse matrix
		:return: the posterior belief vector of products and users
		'''

		# transfer to sparse row matrix to accelerate calculation
		self.pu_csr_matrix = self.pu_matrix.tocsr()
		self.diag_pu_csr_matrix = self.diag_pu_matrix.tocsr()

		# # different normalization approaches: D^(-1/2)*A*D^(-1/2)
		# D_12 = self.diag_pu_csr_matrix.power(-1/2)
		# D_1 = self.diag_pu_csr_matrix.power(-1)
		# self.nor_pu_csr_matrix = D_12 * self.pu_csr_matrix * D_1


		# without normalization
		self.nor_pu_csr_matrix = self.pu_csr_matrix

		i = 0
		while i < max_iters:
			sum_0 = np.sum(self.res_pu_spam_post_vector)# + np.sum(self.res_pu_benign_post_vector)
			self.res_pu_spam_post_vector = self.res_pu_spam_prior_vector + 2 * 0.008 * (self.nor_pu_csr_matrix.dot(self.res_pu_spam_post_vector))
			# self.res_pu_benign_post_vector = self.res_pu_benign_prior_vector + 2 * 0.008 * (self.pu_csr_matrix.dot(self.res_pu_benign_post_vector))
			sum_1 = np.sum(self.res_pu_spam_post_vector)# + np.sum(self.res_pu_benign_post_vector)

			# print('iter: ' + str(i))
			# print('diff: ' + str(abs(sum_0 - sum_1)))

			i += 1

			if abs(sum_0 - sum_1) < 0.1:
				return abs(sum_0 - sum_1)

	def pur_lbp(self):
		'''
		Run the matrix form of lbp on the product-user-review sparse matrix
		:return: the posterior belief vector of all nodes
		'''
		i = 0
		self.pu_csr_matrix = self.pu_matrix.tocsr()
		while i < self.max_iters:
			sum_0 = np.sum(self.res_pur_post_vector)

			self.res_pur_post_vector = self.res_pur_prior_vector + 2 * 0.01 * (self.pur_matrix.dot(self.res_pur_post_vector))
			sum_1 = np.sum(self.res_pur_post_vector)

			# print('iter: ' + str(i))
			# print('diff: ' + str(abs(sum_0 - sum_1)))

			i += 1

			if abs(sum_0 - sum_1) < 0.1:
				return self.res_pur_post_vector

	def classify(self):
		'''
		Calculate the posterior belief of three type of nodes
		:return: u_post: users posterior beliefs, p_post: products posterior beliefs,
		r_post: reviews posterior beliefs.
		'''
		u_post = {}
		p_post = {}
		r_post = {}
		pu_post = self.res_pu_spam_post_vector#/(self.res_pu_benign_post_vector + self.res_pu_spam_post_vector)
		no_prod = len(self.p_priors)
		# extract the posterior belief of users and reviews
		for i, r in enumerate(pu_post[no_prod:]):
			u_post[str(i + no_prod)] = float(r)
		for i, r in enumerate(pu_post[:no_prod]):
			p_post[str(i)] = float(r)
		for i, r in self.r_priors.items():
			r_post[i] = (u_post[i[0]] + float(r)) / 2

		u_post = scale_value(u_post)
		p_post = scale_value(p_post)
		r_post = scale_value(r_post)

		return u_post, p_post, r_post

if __name__ == '__main__':

	# dataset source
	feature_suspicious_filename = 'feature_configuration.txt'

	tencent_filename = 'data_0728.gz'
	YelpChi_filename = 'Yelp_Dataset/YelpChi/metadata.gz'
	YelpNYC_filename = 'Yelp_Dataset/YelpChi/metadata.gz'

	# tencent feature map
	# review_feature_list = ['RD', 'EXT', 'EXT', 'DEV']
	# user_feature_list = ['MNR', 'PR', 'NR', 'ETG']
	# product_feature_list = ['MNR', 'PR', 'NR', 'ETG']
	# yelp feature map
	review_feature_list = ['RD', 'EXT', 'EXT', 'DEV', 'ETF', 'ISR']
	user_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'BST', 'ERD', 'ETG']
	product_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'ERD', 'ETG']

	# read the graph and set parameters
	user_product_graph, product_user_graph = read_graph_data(YelpChi_filename)
	feature_config = load_feature_config(feature_suspicious_filename)

	# construct the features based on the original graph then update with the new graph
	feature_extractor = FeatureExtractor()
	UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.construct_all_features(user_product_graph, product_user_graph)
	new_upriors = feature_extractor.calculateNodePriors(user_feature_list, UserFeatures, feature_config)
	new_ppriors = feature_extractor.calculateNodePriors(product_feature_list, ProdFeatures, feature_config)
	new_rpriors = feature_extractor.calculateNodePriors(review_feature_list, ReviewFeatures, feature_config)

	priors = [new_upriors, new_ppriors, new_rpriors]

	# calculate the max/mean/min value of the prior of each node type
	ranked_upriors = [(user, new_upriors[user]) for user in new_upriors.keys()]
	ranked_upriors = sorted(ranked_upriors, reverse=True, key=lambda x: x[1])
	ranked_rpriors = [(user, new_rpriors[user]) for user in new_rpriors.keys()]
	ranked_rpriors = sorted(ranked_rpriors, reverse=True, key=lambda x: x[1])
	ranked_ppriors = [(user, new_ppriors[user]) for user in new_ppriors.keys()]
	ranked_ppriors = sorted(ranked_ppriors, reverse=True, key=lambda x: x[1])
	u_max, u_mean, u_min = ranked_upriors[0][1], ranked_upriors[int(len(ranked_upriors) / 2)][1], ranked_upriors[-1][1]
	p_max, p_mean, p_min = ranked_ppriors[0][1], ranked_ppriors[int(len(ranked_ppriors) / 2)][1], ranked_ppriors[-1][1]
	r_max, r_mean, r_min = ranked_rpriors[0][1], ranked_rpriors[int(len(ranked_rpriors) / 2)][1], ranked_rpriors[-1][1]

	# centering the priors to 0.5
	# u_diff = 0.5 - u_mean
	# p_diff = 0.5 - p_mean
	# r_diff = 0.5 - r_mean
	#
	# for i, p in priors[0].items():
	# 	priors[0][i] = p + u_diff
	# for i, p in priors[1].items():
	# 	priors[1][i] = p + p_diff
	# for i, p in priors[2].items():
	# 	priors[2][i] = p + r_diff
	#
	# # centering the priors to 0.5
	# new_upriors, new_ppriors, new_rpriors = priors[0], priors[1], priors[2]
	# ranked_upriors = [(user, new_upriors[user]) for user in new_upriors.keys()]
	# ranked_upriors = sorted(ranked_upriors, reverse=True, key=lambda x: x[1])
	# ranked_rpriors = [(user, new_rpriors[user]) for user in new_rpriors.keys()]
	# ranked_rpriors = sorted(ranked_rpriors, reverse=True, key=lambda x: x[1])
	# ranked_ppriors = [(user, new_ppriors[user]) for user in new_ppriors.keys()]
	# ranked_ppriors = sorted(ranked_ppriors, reverse=True, key=lambda x: x[1])
	# u_max, u_mean, u_min = ranked_upriors[0][1], ranked_upriors[int(len(ranked_upriors) / 2)][1], ranked_upriors[-1][1]
	# p_max, p_mean, p_min = ranked_ppriors[0][1], ranked_ppriors[int(len(ranked_ppriors) / 2)][1], ranked_ppriors[-1][1]
	# r_max, r_mean, r_min = ranked_rpriors[0][1], ranked_rpriors[int(len(ranked_rpriors) / 2)][1], ranked_rpriors[-1][1]

	# normalize the node priors to (0,1)
	# if we normalize the prior, we need to set nor_flg to True for the gang model
	for i, p in priors[0].items():
		priors[0][i] = (p-u_min)/(u_max-u_min)
	for i, p in priors[1].items():
		priors[1][i] = (p-p_min)/(p_max-p_min)
	for i, p in priors[2].items():
		priors[2][i] = (p-r_min)/(r_max-r_min)
	#
	# new_upriors, new_ppriors, new_rpriors = priors[0], priors[1], priors[2]
	# ranked_upriors = [(user, new_upriors[user]) for user in new_upriors.keys()]
	# ranked_upriors = sorted(ranked_upriors, reverse=True, key=lambda x: x[1])
	# ranked_rpriors = [(user, new_rpriors[user]) for user in new_rpriors.keys()]
	# ranked_rpriors = sorted(ranked_rpriors, reverse=True, key=lambda x: x[1])
	# ranked_ppriors = [(user, new_ppriors[user]) for user in new_ppriors.keys()]
	# ranked_ppriors = sorted(ranked_ppriors, reverse=True, key=lambda x: x[1])
	# u_max, u_mean, u_min = ranked_upriors[0][1], ranked_upriors[int(len(ranked_upriors) / 2)][1], ranked_upriors[-1][1]
	# p_max, p_mean, p_min = ranked_ppriors[0][1], ranked_ppriors[int(len(ranked_ppriors) / 2)][1], ranked_ppriors[-1][1]
	# r_max, r_mean, r_min = ranked_rpriors[0][1], ranked_rpriors[int(len(ranked_rpriors) / 2)][1], ranked_rpriors[-1][1]
	# exit()

	mean_priors = [0.5, 0.5, 0.5]

	# create ground truth
	user_ground_truth, review_ground_truth = create_ground_truth(user_product_graph)

	# run lbp iteratively
	# for max_iter in range(1, 50):
	model = gang(user_product_graph, product_user_graph, user_ground_truth, review_ground_truth,
				 priors, mean_priors, 0.1, nor_flg=True, sup_flg=False)

	# # run lbp on the product-user-review matrix
	# res_pur_post = model.pur_lbp()
	#
	# u_post = {}
	# r_post = {}
	# pur_post = res_pur_post
	#
	# # extract the posterior belief of users and reviews
	# no_prod = len(priors[1])
	# no_user = len(priors[0])
	# for i, r in enumerate(pur_post[no_prod:no_user+no_prod]):
	# 	u_post[str(i+no_prod)] = r
	# r_id = 0
	# for r in priors[2].keys():
	# 	r_post[r] = pur_post[no_user+no_prod+r_id]
	# 	r_id += 1


	# run lbp on the product-user matrix
	time1 = time.time()
	model.pu_lbp(2000)
	time2 = time.time()
	print(str(time2 - time1))

	u_post, _, r_post = model.classify()
	# print(u_post)
	# evaluate the model
	user_AUC, user_AP = evaluate(user_ground_truth, u_post)
	review_AUC, review_AP = evaluate(review_ground_truth, r_post)
	print(user_AP)
	print(user_AUC)
	print(review_AP)
	print(review_AUC)

	# visualize result
	# ranked_uBeliefs = [(user, u_post[user]) for user in u_post.keys()]
	# ranked_uBeliefs = sorted(ranked_uBeliefs, reverse=True, key=lambda x: x[1])
	# print(sum([priors[1] for priors in ranked_uBeliefs])/len(user_ground_truth.keys()))
	# print(ranked_uBeliefs[math.ceil(len(user_ground_truth.keys())/2)])
	#
	# for item in ranked_uBeliefs[:20]:
	# 	print(str(item[0]) + ' ' + str(len(user_product_graph[item[0]])) + ' ' + str(item[1]))
	# for item in ranked_uBeliefs[-100:]:
	# 	print(str(item[0]) + ' ' + str(len(user_product_graph[item[0]])) + ' ' + str(item[1]))
	#
	# no_slot = math.ceil(len(user_ground_truth.keys())/500)
	# x = [i for i in range(0, no_slot)]
	# y1 = [0 * i for i in range(0, no_slot)]
	# y2 = [0 * i for i in range(0, no_slot)]
	# y3 = [0 * i for i in range(0, no_slot)]
	# y4 = [0 * i for i in range(0, no_slot)]
	# for i in range(0, no_slot-1):
	# 	for y in range(500*i, 500*(i+1)):
	# 		if y == len(user_ground_truth):
	# 			break
	# 		y1[i] += 1 if user_ground_truth[ranked_uBeliefs[y][0]] == 1 else 0
	# 		y2[i] += 1 if user_ground_truth[ranked_uBeliefs[y][0]] == 0 else 0
	# 		y3[i] += 1 if len(user_product_graph[ranked_uBeliefs[y][0]]) <= 2 else 0
	# 		y4[i] += 1 if len(user_product_graph[ranked_uBeliefs[y][0]]) > 2 else 0
	#
	# plt.figure(3)
	# p3 = plt.bar(x, y4)
	# p4 = plt.bar(x, y3, bottom=y4)
	# plt.title('edge distribution of posteriors')
	# plt.yticks(np.arange(0, 500, 100))
	# plt.legend((p4[0], p3[0]), ('#degree=<2', '#degree>2'))
	# plt.show()

	# plt.figure(4)
	# p1 = plt.bar(x, y2)
	# p2 = plt.bar(x, y1, bottom=y2)
	# plt.title('label distribution of posteriors')
	# plt.yticks(np.arange(0, 500, 100))
	# plt.legend((p2[0], p1[0]), ('Spam', 'Benign'))
	# plt.show()

	# normalize user posterior
	ranked_upost = [(user, u_post[user]) for user in u_post.keys()]
	ranked_upost = sorted(ranked_upost, reverse=True, key=lambda x: x[1])

	up_max, up_mean, up_min = ranked_upost[0][1], ranked_upost[int(len(ranked_upost) / 2)][1], ranked_upost[-1][1]

	nor_upost = {}
	for i, p in u_post.items():
		nor_upost[i] = (p-up_min)/(up_max-up_min)

	# posterior of all account nodes with respect to degree
	x = np.zeros(len(user_product_graph))
	y = np.zeros(len(user_product_graph))
	for iter, user in enumerate(user_product_graph):
		x[iter] = np.log(len(user_product_graph[user]))
		y[iter] = np.log(nor_upost[user])

	plt.figure(1)
	p1 = plt.scatter(x, y, marker='o', c='r')
	plt.title('GANG')
	plt.xlabel('Log of node degrees')
	plt.ylabel('Log of node posteriors')
	plt.show()