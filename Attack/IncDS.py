import copy
import time


"""
	The implementation of the IncDS attack.
"""


def compute_density(user_product_graph, product_user_graph, c, t):
	"""
	Compute the density of controlled accounts according to their local structural density
	"""

	density = {}
	# intialize the auxiliary graph
	aux_user_graph = copy.deepcopy(user_product_graph)
	aux_prod_graph = copy.deepcopy(product_user_graph)
	for u in c:
		aux_user_graph[u].append((t, 1, -1, '2012-06-01'))
		aux_prod_graph[t].append((u, 1, -1, '2012-06-01'))
	for u in c:
		user_degree = len(aux_user_graph[u])
		prod_degree = sum([len(aux_prod_graph[review[0]]) for review in aux_user_graph[u]])
		density[u] = user_degree/prod_degree

	return density

def ds_evasion(user_product_graph, product_user_graph, c, r, t):
	"""
	Args:
		user_product_graph: key = user_id, value = list of review tuples
		product_product_graph: key = product_id, value = list of review tuples
		priors: node priors
		c: list of controlled accounts
		r: number of reviews to be posted each account
		t: target list
		feature_config:
	"""
	# total number of spams posted
	count = 0
	added_edges = []
	t0 = time.time()

	# how many new controlled accounts are selected to post spams for the current iteration
	unique = 0

	new_user_graph = copy.deepcopy(user_product_graph)
	new_product_graph = copy.deepcopy(product_user_graph)

	account_log = []

	# for each target, find controlled accounts to post spams
	for target in t:

		# compute the density
		density = compute_density(new_user_graph, new_product_graph, c, target)

		selected_accounts = [(account, density[account]) for account in c]

		selected_accounts = sorted(selected_accounts, reverse=False, key=lambda x: x[1])
		print("Dict of densities of controlled accounts")
		print(selected_accounts)

		selected_accounts = [account[0] for account in selected_accounts[:r]]

		print("List of selected accounts")
		print(selected_accounts)

		# count the number of unique added accounts
		for account in selected_accounts:
			if account not in account_log:
				unique += 1
		print('Total number of selected unique accounts: %d' % (unique))

		account_log = account_log + selected_accounts

		# add the added_edges to the global graph
		for added_account in selected_accounts:
			new_user_graph[added_account].append((target, 1, -1, '2012-06-01'))
			new_product_graph[target].append((added_account, 1, -1, '2012-06-01'))

		# add new nodes to output
		for added_account in selected_accounts:
			review_id = (added_account, target)
			added_edges.append(review_id)

		t1 = time.time()

		print('Time consumed: %s' % str(t1 - t0))
		print('\n---------------------------------\n')

	return added_edges, user_product_graph