import random as rd
import sys
import os
sys.path.insert(0, os.path.abspath('../'))

from Utils.iohelper import *


"""
	The implementation of the Random attack.
"""


def random_post(c, t, r, p):
	"""
	c: the list of the controlled accounts
	t: the list of the target businesses
	r: the number of reviews posted per target
	p: the metadata file path prefix 
	"""

	prefix = p
	metadata_filename = prefix + 'metadata.gz'

	# load the graph
	user_product_graph, _ = read_graph_data(metadata_filename)

	added_edges = []
	unique = 0
	account_log =[]

	# random adding edges
	for target in t:

		selected_spammers = rd.sample(c ,r)
		for spammer in selected_spammers:
			user_product_graph[spammer].append((target, 1, -1, '2012-06-01'))
			added_edges.append((spammer, target))

		for account in selected_spammers:
			if account not in account_log:
				unique += 1
		print('Total number of selected unique accounts:: %d' %(unique))

		account_log = account_log + selected_spammers

	return added_edges, user_product_graph
