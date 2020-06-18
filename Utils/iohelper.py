import gzip


"""
	Define several functions to handle files.
"""


def read_graph_data(metadata_filename):
	""" Read the user-review-product graph from file. Can output the graph in different formats
		Args:
			metadata_filename: a gzipped file containing the graph.
		Return:
			graph: user-review / prod-review
	"""

	user_data = {}

	prod_data = {}

	# use the rt mode to read ascii strings instead of binary
	with gzip.open(metadata_filename, 'rt') as f:
		# file format: each line is a tuple (user id, product id, rating, label, date)
		for line in f:
			items = line.strip().split()
			u_id = items[0]
			p_id = items[1]
			rating = float(items[2])
			label = int(items[3])
			date = items[4]

			if u_id not in user_data:
				user_data[u_id] = []
			user_data[u_id].append((p_id, rating, label, date))

			if p_id not in prod_data:
				prod_data[p_id] = []
			prod_data[p_id].append((u_id, rating, label, date))

	# delete reviews from graph to create new products
	if 'NYC' in metadata_filename:
		user_data, prod_data = create_new_products(user_data, prod_data, (0, 120))

	# print('read reviews from %s' % metadata_filename)
	# print('number of users = %d' % len(user_data))
	# print('number of products = %d' % len(prod_data))

	return user_data, prod_data


def load_feature_config(config_filename):
	"""
	Read configuration about how the value of a feature indicates suspiciousness of a node
	"""
	config = {}

	f = open(config_filename, 'r')
	for line in f:
		if line[0] == '+' or line[0] == '-':
			# print (line)
			items = line.split(' ')
			direction = items[0].strip(':')
			feature_name = items[1]
			config[feature_name] = direction
	f.close()

	return config


def create_new_products(user_data, prod_data, range):
	"""
	Special pre-processing for the YelpNYC since there is no new products (#reviews<5)
	:param range: the range of products to be processed
	"""

	product_list = [(product, len(user)) for (product, user) in prod_data.items()]
	sorted_product_list = sorted(product_list, reverse=False, key=lambda x: x[1])
	new_products = [product[0] for product in sorted_product_list[range[0]:range[1]]]
	for item in new_products:
		for review in prod_data[item][1:]:
			for r in user_data[review[0]]:
				if r[0] == item:
					user_data[review[0]].remove(r)
			if len(user_data[review[0]]) == 0:
				user_data.pop(review[0])
		prod_data[item] = [prod_data[item][0]]

	# renumbering the graphs
	start_no = len(prod_data)
	index = {}
	for user in user_data.keys():
		if user not in index.keys():
			index[user] = str(start_no)
			start_no += 1
	new_user_data = {}
	new_prod_data = {}
	for user, reviews in user_data.items():
		new_user_data[index[user]] = reviews
	for prod, reviews in prod_data.items():
		new_prod_data[prod] = []
		for review in reviews:
			new_prod_data[prod].append((index[review[0]], review[1], review[2], review[3]))

	return new_user_data, new_prod_data
