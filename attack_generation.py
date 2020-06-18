import pickle
import sys
sys.path.insert(0, sys.path[0] + '/..')

from Utils.iohelper import *
from Attack.IncBP import bp_evasion
from Attack.IncDS import ds_evasion
from Attack.IncPR import pr_evasion
from Attack.Random import random_post
from Utils.yelpFeatureExtraction import *


"""
	Generating reviews for different spamming attack strategies.
"""


if __name__ == '__main__':

	# initialize the attack setting
	dataset = 'YelpChi'  # YelpChi, YelpNYC, YelpZip
	attack = 'IncBP'  # IncBP, IncDS, IncPR, Random, Singleton
	prefix = 'Yelp_Dataset/' + dataset + '/'
	metadata_filename = prefix + 'metadata.gz'
	exp_setting = {'YelpChi': (100, 30), 'YelpNYC': (400, 120), 'YelpZip': [700, 600]}
	accounts, targets = exp_setting[dataset]

	print('Executing {} attack on {}...'.format(attack, dataset))

	# load the graph
	user_product_graph, product_user_graph = read_graph_data(metadata_filename)

	# feature and prior calculation
	feature_suspicious_filename = 'Utils/feature_configuration.txt'
	review_feature_list = ['RD', 'EXT', 'EXT', 'DEV', 'ETF', 'ISR']
	user_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'BST', 'ERD', 'ETG']
	product_feature_list = ['MNR', 'PR', 'NR', 'avgRD', 'ERD', 'ETG']
	feature_config = load_feature_config(feature_suspicious_filename)
	feature_extractor = FeatureExtractor()
	UserFeatures, ProdFeatures, ReviewFeatures = feature_extractor.construct_all_features(user_product_graph,
																						  product_user_graph)
	new_upriors = feature_extractor.calculateNodePriors(user_feature_list, UserFeatures, feature_config)
	new_ppriors = feature_extractor.calculateNodePriors(product_feature_list, ProdFeatures, feature_config)
	new_rpriors = feature_extractor.calculateNodePriors(review_feature_list, ReviewFeatures, feature_config)
	priors = [new_upriors, new_rpriors, new_ppriors]

	# generate product ranking based on their links
	product_list = [(product, len(user)) for (product, user) in product_user_graph.items()]
	sorted_product_list = sorted(product_list, reverse=True, key=lambda x: x[1])

	# generate user ranking based on their priors
	user_list = [(user, new_upriors[user]) for user in user_product_graph.keys()]
	sorted_user_list = sorted(user_list, reverse=True, key=lambda x: x[1])

	# select the set of target products
	new_businesses = [product[0] for product in sorted_product_list[-targets:]]

	# select the set of controlled elite accounts
	elite_spammers = []
	for user in reversed(sorted_user_list):
		labels = [review[2] for review in user_product_graph[user[0]]]
		if -1 in labels:  # don't select suspicious accounts
			continue
		else:
			if len(user_product_graph[user[0]]) >= 10:
				elite_spammers.append(user[0])
		if len(elite_spammers) == accounts:
			break

	r = 15  # number of reviews per target
	c = elite_spammers  # controlled accounts
	t = new_businesses  # target products

	if attack == 'IncBP':
		added_edges, _, _ = bp_evasion(user_product_graph, product_user_graph, priors, c, r, t, feature_config)
	elif attack == 'IncDS':
		added_edges, _ = ds_evasion(user_product_graph, product_user_graph, c, r, t)
	elif attack == 'IncPR':
		added_edges, _ = pr_evasion(user_product_graph, product_user_graph, c, r, t, feature_config)
	elif attack == 'Random':
		added_edges, _ = random_post(c, t, r, prefix)
	elif attack == 'Singleton':
		added_edges = []
		for user in user_product_graph.keys():
			new_id = int(user) + 1
		c = [str(user) for user in range(new_id, new_id + targets*r)]
		for index, prod in enumerate(t):
			for account in c[r * index:r * (index+1)]:
				added_edges.append((account, prod))

	print(len(added_edges))

	# save the generated reviews for corresponding attack
	with open('Testing/' + dataset + '/' + attack + '.pickle', 'wb') as f:
		pickle.dump([c, t, added_edges], f)
	f.close()
