import copy as cp
import pickle
import random as rd

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from Smoother_Fraudar import *
from eval_GANG import *
from eval_SpEagle import *
from eval_fBox import *
from eval_helper import *
from iohelper import *
from scipy.special import expit
from yelpFeatureExtraction import *


def e_greedy_sample(attacks_p, e=.2):
	"""
	e-greedy algorithm for sampling attack strategies from discrete distributions
	"""

	z = np.random.random()

	attacks_p.keys()

	if z > e: # sample attack based on their probabilities
		attack = np.random.choice(list(attacks_p), 1, p=list(attacks_p.values()))
		attack = attack[0]
	else: # random sample
		attack = rd.choice(list(attacks_p))

	return attack


def load_fake_reviews(attacks, targets, paths, attack_para):
	"""
	load fake reviews for each attack to each item
	"""

	attack_item_review = {}


	for item in targets:
		attack_reviews = {}
		for attack in attacks:
			attack_reviews[attack] = []
			with open(paths[attack], 'rb') as f:
				evasions = pickle.load(f)
			if attack is 'Singleton':
				# print(evasions)
				# print(paths[attack])
				# print(evasions[1])
				item_index = evasions[1].index(item)
				for i in range(30*item_index, 30*item_index+15):
					attack_reviews[attack].append((evasions[0][i], item))
			else:
				for index, review in enumerate(evasions[2][:attack_para[1]]):
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
	gang_model, _ = runGANG(new_priors, u_p_graph, p_u_graph, user_ground_truth, review_ground_truth)
	gang_ubelief, _, gang_rbelief = gang_model.classify()

	print('Run SpEagle ...')
	# get posteriors from SpEagle
	speagle_model, _ = runSpEagle(new_priors, u_p_graph)
	speagle_ubelief, speagle_rbelief, _ = speagle_model.classify()

	print('Run Fraudar ...')
	# get posteriors from Fraudar
	fraudar_ubelief, fraudar_rbelief = runNew(new_priors, u_p_graph, p_u_graph, user_ground_truth, review_ground_truth)

	print('Run fBox ...')
	# get posteriors from fBox
	_, fbox_ubelief, fbox_rbelief = runfBox(new_priors, u_p_graph)

	print('Run Prior ...')
	# get posteriors from Prior
	prior_ubelief, prior_rbelief = new_priors[3], new_priors[4]

	# normalize the output
	speagle_rbelief, fraudar_rbelief, fbox_rbelief, prior_rbelief = scale_value(speagle_rbelief), scale_value(fraudar_rbelief), scale_value(fbox_rbelief), scale_value(prior_rbelief)


	print('GANG: {}, SpEagle: {}, Fraudar: {}, fBox: {}, Prior: {}'.format(
		(max(gang_rbelief.values()), min(gang_rbelief.values())), (max(speagle_rbelief.values()), min(speagle_rbelief.values())),
	(max(fraudar_rbelief.values()), min(fraudar_rbelief.values())), (max(fbox_rbelief.values()), min(fbox_rbelief.values())),
	(max(prior_rbelief.values()), min(prior_rbelief.values()))))

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
		r_spam_beliefs[r] = expit(accu_spam_belief) # 1 / (1 + math.exp(-accu_belief))
		r_detector_belief[r] = {'GANG': gang_rbelief[r], 'SpEagle': speagle_rbelief[r], 'Fraudar': fraudar_rbelief[r], 'fBox': fbox_rbelief[r], 'Prior': prior_rbelief[r]}

	# rank the top_k suspicious reviews
	ranked_rbeliefs = [(review, r_spam_beliefs[review]) for review in r_spam_beliefs.keys()]
	ranked_rbeliefs = sorted(ranked_rbeliefs, reverse=True, key=lambda x: x[1])
	top_k = int(len(r_spam_beliefs) * top_k_per)
	top_k_reviews = [review[0] for review in ranked_rbeliefs[:top_k]]
	print(len(top_k_reviews))
	# remove false positives
	for review in top_k_reviews:
		if review not in new_added_reviews:
			top_k_reviews.remove(review)

	print('top k is {}'.format(len(top_k_reviews)))

	return r_spam_beliefs, r_accu_spam_beliefs, r_detector_belief, top_k_reviews, new_priors, spammer_ids, u_p_graph, p_u_graph, new_added_reviews


def compute_re(user_product_graph, product_user_graph, targets, elite):
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
			if len(user_product_graph[review[0]]) >= elite:
				temp_ERI.append(review[1])
		ERI[target] = sum(temp_ERI)/len(temp_ERI) if len(temp_ERI) != 0 else 0

	# Eq. (1) in paper
	for target in targets:
		Revenue[target] = 0.09 + 0.035 * RI[target] + 0.036 * ERI[target]

	return RI, ERI, Revenue


def remove_topk_reviews(user_product_graph, product_user_graph, filter_reviews):
	"""
	Remove top_k suspicious reviews from the given review graph
	"""

	for edge in filter_reviews:
		for review in product_user_graph[edge[1]]:
			if review[0] == edge[0]:
				product_user_graph[edge[1]].remove(review)
		for review in user_product_graph[edge[0]]:
			if review[0] == edge[1]:
				user_product_graph[edge[0]].remove(review)

	return user_product_graph, product_user_graph


def compute_cost(ori, new, added_reviews, filter_reviews, target_ids, new_user_product_graph, elite):
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
		if len(new_user_product_graph[review[0]]) >= elite:
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
		# reward[item] = math.tanh((PE - avg_PE) / PE_interval)
	return reward


def update_p(attacks_p, item_attack_mapping, reward, lr2):
	"""
	Updating the sampling probabilities of attacks according to the rewards computed before
	:param lr2: the learning rate of attacks_p
	:return:
	"""

	# record the selection times and reward of attack during each update
	counter = {a:0 for a in attacks_p.keys()}
	reward_log = cp.deepcopy(counter)

	# update the P with normalized reward
	for i, r in reward.items():
		current_attack = item_attack_mapping[i]
		# current_count, current_reward = counter[current_attack], reward_log[current_attack]

		# compute the new reward using the discount factor
		# new_reward = current_reward + 1/(current_count+1) * (r - current_reward)
		new_reward = r
		# update the P with new reward
		attacks_p[current_attack] += lr2 * new_reward

		# update counter and reward log
		# counter[current_attack] += 1
		# reward_log[current_attack] = new_reward

	for a, p in attacks_p.items():
		if p <= 0:
			attacks_p[a] = 0

	return attacks_p


def update_q(detectors_q, cost, r_detector_belief, r_accu_spam_beliefs, remain_reviews, lr1):
	"""
	Updating the weights of detectors based on their gradients of the cost-sensitive loss function
	"""

	grad_log = {d:0 for d in detectors_q.keys()}
	counter = cp.deepcopy(grad_log)

	# the gradient of Eq. (10) in paper
	for d, q in detectors_q.items():
		grad_sum = 0
		for review in remain_reviews:
			# calculate gradient for current detector
			grad_sum += -1 * cost[review] * r_detector_belief[review][d] * expit(-r_accu_spam_beliefs[review])

		grad_norm = grad_sum / len(remain_reviews)
		# update detector weight
		detectors_q[d] = q - lr1 * grad_norm

	return detectors_q


def compute_total(ori_Revenue, new_Revenue, targets, cost, r_spam_beliefs):
	"""
	Compute total loss of detectors and total practical effort of attacks
	"""

	total_pe = sum([new_Revenue[i]-ori_Revenue[i] for i in targets])

	total_loss = sum([-1 * cost[r] * np.log(r_spam_beliefs[r]) for r in cost.keys()])/len(cost)

	return total_pe, total_loss


def plot_result(attack_log, detector_log, loss_log, pe_log, remain_reviews_log, attack_para):
	"""
	Plot the results
	"""

	attack_label = list(attack_log[0].keys())
	detector_label = list(detector_log)
	colors = {'GANG': '#cc66cc', 'SpEagle': '#3288bd', 'fBox': '#fc8d59', 'Fraudar': '#99d594', 'Prior': '#fee08b'}
	marker_style = {'GANG': 'o', 'SpEagle': 'd', 'fBox': 'v', 'Fraudar': '*', 'Prior': 'x'}
	iter = [i for i in range(50 + 1)]

	# mean = {'IncBP': [], 'IncDS': [], 'IncPR': [], 'Random': [], 'Singleton': []}
	# upper = cp.deepcopy(mean)
	# lower = cp.deepcopy(mean)

	# for a, log in attack_log[0].items():
	# 	log1 = attack_log[1][a]
	# 	log2 = attack_log[2][a]
	# 	for i, v in enumerate(log):
	# 		ranked = sorted([v, log1[i], log2[i]])
	# 		lower[a].append(ranked[0])
	# 		mean[a].append((ranked[0] + ranked[2])/2.0)
	# 		upper[a].append(ranked[2])

	# fig = plt.figure(1)
	# gs = gridspec.GridSpec(1, 3, figure=fig)
	# ax1 = fig.add_subplot(gs[0, 0])
	# for a, log in mean.items():
	# 	ax1.plot(iter, [0.2] + log, linewidth=1.5, label=a)
	# 	ax1.fill_between(iter, [0.2] + lower[a], [0.2] + upper[a], alpha=0.3)
	
	# ax1.set_xlabel('Episode', fontsize='15')
	# ax1.set_ylabel('Attack Sampling Probability', fontsize='15')
	# ax1.set_xticks(np.arange(0, len(iter), 10))
	# plt.title('YelpChi', fontsize='20')
	# ax1.set_ylim(0, 0.42)
	# # ax1.set_yticks(np.arange(0, 0.7, 0.1))
	# ax1.legend(loc='upper left', ncol=2, prop={'size':12}, frameon=False, labelspacing=0.01, handlelength=1, handletextpad=0.1, borderaxespad=0.01)

	# mean = {'IncBP': [], 'IncDS': [], 'IncPR': [], 'Random': [], 'Singleton': []}
	# upper = cp.deepcopy(mean)
	# lower = cp.deepcopy(mean)

	# for a, log in attack_log[3].items():
	# 	log1 = attack_log[4][a]
	# 	log2 = attack_log[5][a]
	# 	for i, v in enumerate(log):
	# 		ranked = sorted([v, log1[i], log2[i]])
	# 		lower[a].append(ranked[0])
	# 		mean[a].append((ranked[0] + ranked[2])/2.0)
	# 		upper[a].append(ranked[2])

	# ax2 = fig.add_subplot(gs[0, 1])
	# for a, log in mean.items():
	# 	ax2.plot(iter, [0.2] + log, linewidth=1.5, label=a)
	# 	ax2.fill_between(iter, [0.2] + lower[a], [0.2] + upper[a], alpha=0.3)
	# ax2.set_xlabel('Episode', fontsize='15')
	# # ax2.set_ylabel('Attack Sampling Probability', fontsize='15', fontweight='bold')
	# ax2.set_xticks(np.arange(0, len(iter), 10))
	# plt.title('YelpNYC', fontsize='20')
	# ax2.set_ylim(0, 0.42)
	# ax2.set_yticks([])

	# mean = {'IncBP': [], 'IncDS': [], 'IncPR': [], 'Random': [], 'Singleton': []}
	# upper = cp.deepcopy(mean)
	# lower = cp.deepcopy(mean)

	# for a, log in attack_log[6].items():
	# 	log1 = attack_log[7][a]
	# 	log2 = attack_log[8][a]
	# 	for i, v in enumerate(log):
	# 		ranked = sorted([v, log1[i], log2[i]])
	# 		lower[a].append(ranked[0])
	# 		mean[a].append((ranked[0] + ranked[2])/2.0)
	# 		upper[a].append(ranked[2])

	# ax3 = fig.add_subplot(gs[0, 2])
	# for a, log in mean.items():
	# 	ax3.plot(iter, [0.2] + log, linewidth=1.5, label=a)
	# 	ax3.fill_between(iter, [0.2] + lower[a], [0.2] + upper[a], alpha=0.3)
	# ax3.set_xlabel('Episode', fontsize='15')
	# # ax3.set_ylabel('Attack Sampling Probability', fontsize='15', fontweight='bold')
	# ax3.set_xticks(np.arange(0, len(iter), 10))
	# plt.title('YelpZip', fontsize='20')
	# ax3.set_ylim(0, 0.42)
	# ax3.set_yticks([])


	# plt.subplots_adjust(bottom=0.15, wspace=0.05)
	# plt.show()

	# mean = {'SpEagle': [], 'Fraudar': [], 'GANG': [], 'Prior': [], 'fBox': []}
	# upper = cp.deepcopy(mean)
	# lower = cp.deepcopy(mean)

	# for a, log in detector_log[0].items():
	# 	log1 = detector_log[1][a]
	# 	log2 = detector_log[2][a]
	# 	for i, v in enumerate(log):
	# 		ranked = sorted([v, log1[i], log2[i]])
	# 		lower[a].append(ranked[0])
	# 		mean[a].append((ranked[0] + ranked[2])/2.0)
	# 		upper[a].append(ranked[2])

	# # print(lower, upper)

	# fig = plt.figure(2)
	# markers = list(np.arange(0, len(iter), 25))
	# print(markers)
	# gs = gridspec.GridSpec(1, 3, figure=fig)
	# ax1 = fig.add_subplot(gs[0, 0])
	# for d, log in mean.items():
	# 	ax1.plot(iter, [0.2] + log[:300], linewidth=3, label=d, markevery=markers, color=colors[d], marker=marker_style[d], markerfacecolor='white', markersize=10)
	# 	# ax1.fill_between(iter, [0.2] + lower[d], [0.2] + upper[d], alpha=1)
	# ax1.set_xlabel('Episode', fontsize='15')
	# ax1.set_ylabel('Detector Weight', fontsize='15')
	# plt.title('YelpChi', fontsize='20')
	# ax1.set_xticks(np.arange(0, len(iter), 10))
	# ax1.legend(loc='upper left', ncol=2, prop={'size':9}, columnspacing=1, labelspacing=0.01, handlelength=1, frameon=False, handletextpad=0.1, borderaxespad=0.01)
	# # ax1.set_ylim(0, 0.6)

	# mean = {'SpEagle': [], 'Fraudar': [], 'GANG': [], 'Prior': [], 'fBox': []}
	# upper = cp.deepcopy(mean)
	# lower = cp.deepcopy(mean)

	# for a, log in detector_log[3].items():
	# 	log1 = detector_log[4][a]
	# 	log2 = detector_log[5][a]
	# 	for i, v in enumerate(log):
	# 		ranked = sorted([v, log1[i], log2[i]])
	# 		lower[a].append(ranked[0])
	# 		mean[a].append((ranked[0] + ranked[2])/2.0)
	# 		upper[a].append(ranked[2])

	# ax2 = fig.add_subplot(gs[0, 1])
	# for d, log in mean.items():
	# 	ax2.plot(iter, [0.2] + log[:300], linewidth=3, label=d, markevery=markers, color=colors[d], marker=marker_style[d], markerfacecolor='white', markersize=10)
	# ax2.set_xlabel('Episode', fontsize='15')
	# # ax2.set_ylabel('Detector Weight', fontsize='15', fontweight='bold')
	# plt.title('YelpNYC', fontsize='20')
	# ax2.set_xticks(np.arange(0, len(iter), 10))
	# # ax2.set_ylim(0, 0.6)
	# ax2.set_yticks([])


	# mean = {'SpEagle': [], 'Fraudar': [], 'GANG': [], 'Prior': [], 'fBox': []}
	# upper = cp.deepcopy(mean)
	# lower = cp.deepcopy(mean)

	# for a, log in detector_log[6].items():
	# 	log1 = detector_log[7][a]
	# 	log2 = detector_log[8][a]
	# 	for i, v in enumerate(log):
	# 		ranked = sorted([v, log1[i], log2[i]])
	# 		lower[a].append(ranked[0])
	# 		mean[a].append((ranked[0] + ranked[2])/2.0)
	# 		upper[a].append(ranked[2])


	# ax3 = fig.add_subplot(gs[0, 2])
	# for d, log in mean.items():
	# 	ax3.plot(iter, [0.2] + log[:300], linewidth=3, label=d, markevery=markers, color=colors[d], marker=marker_style[d], markerfacecolor='white', markersize=10)
	# ax3.set_xlabel('Episode', fontsize='15')
	# # ax3.set_ylabel('Detector Weight', fontsize='15', fontweight='bold')
	# plt.title('YelpZip', fontsize='20')
	# ax3.set_xticks(np.arange(0, len(iter), 10))
	# # ax3.set_ylim(0, 0.6)
	# ax3.set_yticks([])


	# plt.subplots_adjust(bottom=0.15, wspace=0.05)

	# plt.show()


	# mean = []
	# upper = cp.deepcopy(mean)
	# lower = cp.deepcopy(mean)

	# for a, log in enumerate(pe_log[0]):
	# 	log1 = pe_log[1][a]
	# 	log2 = pe_log[2][a]
	# 	ranked = sorted([log, log1, log2])
	# 	lower.append(ranked[0])
	# 	mean.append((ranked[0] + ranked[2])/2.0)
	# 	upper.append(ranked[2])

	# fig = plt.figure(3)
	
	# gs = gridspec.GridSpec(1, 3, figure=fig)
	# ax1 = fig.add_subplot(gs[0, 0])
	# ax1.plot(iter[1:], [4.9125 for i in iter[1:]], linewidth=3, color='#fc8d59', label='fBox', marker='v',markerfacecolor='white', markersize=10 , markevery=[0 ,25, 49])
	# ax1.plot(iter[1:], [4.9052 for i in iter[1:]], linewidth=3, color='#3288bd', label='SpEagle', marker='d',markerfacecolor='white', markersize=10 , markevery=[0 ,25, 49])
	# ax1.plot(iter[1:], [4.8962 for i in iter[1:]], linewidth=3, color='#99d594', label='Fraudar', marker='*',markerfacecolor='white', markersize=10 , markevery=[0 ,25, 49])
	# ax1.plot(iter[1:], [4.9099 for i in iter[1:]], linewidth=3, color='#fee08b', label='Prior', marker='s',markerfacecolor='white', markersize=10 , markevery=[0 ,25, 49])
	# ax1.plot(iter[1:], [4.9010 for i in iter[1:]], linewidth=3, color='#cc66cc', label='GANG', marker='o',markerfacecolor='white', markersize=10 , markevery=[0 ,25, 49])
	# ax1.plot(iter[1:], mean, linewidth=2, color='yellowgreen', label='Nash-Detect')
	# ax1.fill_between(iter[1:], lower, upper, facecolor='yellowgreen', alpha=0.3)
	# ax1.set_xlabel('Episode', fontsize='15')
	# ax1.set_ylabel('Practical Effect', fontsize='15')
	# plt.title('YelpChi', fontsize='20')
	# ax1.set_xticks(np.arange(0, len(iter), 10))
	# ax1.legend(loc='lower right', ncol=2, prop={'size':11}, columnspacing=1, labelspacing=0.01, handlelength=1, frameon=False, handletextpad=0.1, borderaxespad=0.01)
	# # ax1.legend(loc='lower right', prop={'weight':'bold', 'size':13})
	
	# mean = []
	# upper = cp.deepcopy(mean)
	# lower = cp.deepcopy(mean)

	# for a, log in enumerate(pe_log[3]):
	# 	log1 = pe_log[4][a]
	# 	log2 = pe_log[5][a]
	# 	ranked = sorted([log, log1, log2])
	# 	lower.append(ranked[0])
	# 	mean.append((ranked[0] + ranked[2])/2.0)
	# 	upper.append(ranked[2])


	# ax2 = fig.add_subplot(gs[0, 1])
	# ax2.plot(iter[1:], mean, linewidth=2, color='yellowgreen', label='Nash-Detect')
	# ax2.fill_between(iter[1:], lower, upper, facecolor='yellowgreen', alpha=0.3)
	# ax2.plot(iter[1:], [27.58 for i in iter[1:]], linewidth=3, color='#fc8d59', label='fBox', marker='v',markerfacecolor='white', markersize=10 , markevery=[0 ,25, 49])
	# ax2.plot(iter[1:], [27.47 for i in iter[1:]], linewidth=3, color='#3288bd', label='SpEagle', marker='d',markerfacecolor='white', markersize=10 , markevery=[0 ,25, 49])
	# ax2.plot(iter[1:], [27.42 for i in iter[1:]], linewidth=3, color='#99d594', label='Fraudar', marker='*',markerfacecolor='white', markersize=10 , markevery=[0 ,25, 49])
	# ax2.plot(iter[1:], [27.46 for i in iter[1:]], linewidth=3, color='#fee08b', label='Prior', marker='s',markerfacecolor='white', markersize=10 , markevery=[0 ,25, 49])
	# ax2.plot(iter[1:], [27.76 for i in iter[1:]], linewidth=3, color='#cc66cc', label='GANG', marker='o',markerfacecolor='white', markersize=10 , markevery=[0 ,25, 49])
	# ax2.set_xlabel('Episode', fontsize='15')
	# # ax2.set_ylabel('Practical Effect', fontsize='15', fontweight='bold')
	# plt.title('YelpNYC', fontsize='20')
	# ax2.set_xticks(np.arange(0, len(iter), 10))
	# # ax2.legend(loc='lower right', prop={'weight':'bold', 'size':13})
	
	
	# mean = []
	# upper = cp.deepcopy(mean)
	# lower = cp.deepcopy(mean)

	# for a, log in enumerate(pe_log[6]):
	# 	log1 = pe_log[7][a]
	# 	log2 = pe_log[8][a]
	# 	ranked = sorted([log, log1, log2])
	# 	lower.append(ranked[0])
	# 	mean.append((ranked[0] + ranked[2])/2.0)
	# 	upper.append(ranked[2])

	# ax3 = fig.add_subplot(gs[0, 2])
	# ax3.plot(iter[1:], mean, linewidth=2, color='yellowgreen', label='Nash-Detect')
	# ax3.fill_between(iter[1:], lower, upper, facecolor='yellowgreen', alpha=0.3)
	# ax3.plot(iter[1:], [110.10 for i in iter[1:]], linewidth=3, color='#fc8d59', label='fBox', marker='v',markerfacecolor='white', markersize=10 , markevery=[0 ,25, 49])
	# ax3.plot(iter[1:], [109.55 for i in iter[1:]], linewidth=3, color='#3288bd', label='SpEagle', marker='d',markerfacecolor='white', markersize=10 , markevery=[0 ,25, 49])
	# ax3.plot(iter[1:], [109.48 for i in iter[1:]], linewidth=3, color='#99d594', label='Fraudar', marker='*',markerfacecolor='white', markersize=10 , markevery=[0 ,25, 49])
	# ax3.plot(iter[1:], [109.41 for i in iter[1:]], linewidth=3, color='#fee08b', label='Prior', marker='s',markerfacecolor='white', markersize=10 , markevery=[0 ,25, 49])
	# ax3.plot(iter[1:], [109.69 for i in iter[1:]], linewidth=3, color='#cc66cc', label='GANG', marker='o',markerfacecolor='white', markersize=10 , markevery=[0 ,25, 49])
	# ax3.set_xlabel('Episode', fontsize='15')
	# # ax3.set_ylabel('Practical Effect', fontsize='15')
	# plt.title('YelpZip', fontsize='20')
	# ax3.set_xticks(np.arange(0, len(iter), 10))
	# # ax3.legend(loc='lower right', prop={'weight':'bold', 'size':13})
	
	
	# # plt.subplots_adjust(bottom=0.15, wspace=0.05)
	# plt.show()


	mean = []
	upper = cp.deepcopy(mean)
	lower = cp.deepcopy(mean)

	for a, log in enumerate(loss_log[0]):
		log1 = loss_log[1][a]
		log2 = loss_log[2][a]
		ranked = sorted([log, log1, log2])
		lower.append(ranked[0])
		mean.append((ranked[0] + ranked[2])/2.0)
		upper.append(ranked[2])


	fig = plt.figure(4)
	gs = gridspec.GridSpec(1, 3, figure=fig)
	ax1 = fig.add_subplot(gs[0, 0])
	ax1.plot(iter[1:], mean[:300], linewidth=2, label='Detector Loss')
	ax1.fill_between(iter[1:], lower[:300], upper[:300], alpha=0.3)
	ax1.set_xlabel('Episode', fontsize='15')
	ax1.set_ylabel('Detection Loss', fontsize='15')
	plt.title('YelpChi', fontsize='20')
	ax1.set_xticks(np.arange(0, len(iter), 10))
	ax1.set_ylim(0, 0.009)
	# ax1.set_yticks([])

	mean = []
	upper = cp.deepcopy(mean)
	lower = cp.deepcopy(mean)

	for a, log in enumerate(loss_log[3]):
		log1 = loss_log[4][a]
		log2 = loss_log[5][a]
		ranked = sorted([log, log1, log2])
		lower.append(ranked[0])
		mean.append((ranked[0] + ranked[2])/2.0)
		upper.append(ranked[2])

	ax2 = fig.add_subplot(gs[0, 1])
	ax2.plot(iter[1:], mean, linewidth=2, label='Detector Loss')
	ax2.fill_between(iter[1:], lower, upper, alpha=0.3)
	ax2.set_xlabel('Episode', fontsize='15')
	# ax2.set_ylabel('Detection Loss', fontsize='15')
	plt.title('YelpNYC', fontsize='20')
	ax2.set_xticks(np.arange(0, len(iter), 10))
	ax2.set_ylim(0, 0.009)
	ax2.set_yticks([])

	mean = []
	upper = cp.deepcopy(mean)
	lower = cp.deepcopy(mean)

	for a, log in enumerate(loss_log[6]):
		log1 = loss_log[7][a]
		log2 = loss_log[8][a]
		ranked = sorted([log, log1, log2])
		lower.append(ranked[0])
		mean.append((ranked[0] + ranked[2])/2.0)
		upper.append(ranked[2])

	ax3 = fig.add_subplot(gs[0, 2])
	ax3.plot(iter[1:], mean, linewidth=2, label='Detector Loss')
	ax3.fill_between(iter[1:], lower, upper, alpha=0.3)
	ax3.set_xlabel('Episode', fontsize='15')
	# ax3.set_ylabel('Detection Loss', fontsize='15')
	plt.title('YelpZip', fontsize='20')
	ax3.set_xticks(np.arange(0, len(iter), 10))
	ax3.set_ylim(0, 0.009)
	ax3.set_yticks([])
	plt.subplots_adjust(bottom=0.15, wspace=0.05)
	plt.show()


	# mean = []
	# upper = cp.deepcopy(mean)
	# lower = cp.deepcopy(mean)

	# for a, log in enumerate([len(i) for i in list(remain_reviews_log[0].values())]):
	# 	log1 = [len(i) for i in list(remain_reviews_log[1].values())][a]
	# 	log2 = [len(i) for i in list(remain_reviews_log[2].values())][a]
	# 	ranked = sorted([log, log1, log2])
	# 	lower.append(ranked[0])
	# 	mean.append((ranked[0] + ranked[2])/2.0)
	# 	upper.append(ranked[2])

	# fig = plt.figure(5)
	# gs = gridspec.GridSpec(1, 3, figure=fig)
	# ax1 = fig.add_subplot(gs[0, 0])
	# ax1.plot(iter, [attack_para[0][1]] + mean[:300], color='orange', linewidth=2)
	# ax1.fill_between(iter[1:], lower[:300], upper[:300], facecolor='orange', alpha=0.3)
	# ax1.set_xlabel('Episode', fontsize='15')
	# ax1.set_ylabel('#Evaded Reviews', fontsize='15')
	# plt.title('YelpChi', fontsize='20')
	# ax1.set_xticks(np.arange(0, len(iter), 10))
	# ax1.set_yticks(np.arange(200, attack_para[0][1] + 50, 50))
	

	# mean = []
	# upper = cp.deepcopy(mean)
	# lower = cp.deepcopy(mean)

	# for a, log in enumerate([len(i) for i in list(remain_reviews_log[3].values())]):
	# 	log1 = [len(i) for i in list(remain_reviews_log[4].values())][a]
	# 	log2 = [len(i) for i in list(remain_reviews_log[5].values())][a]
	# 	ranked = sorted([log, log1, log2])
	# 	lower.append(ranked[0])
	# 	mean.append((ranked[0] + ranked[2])/2.0)
	# 	upper.append(ranked[2])

	# ax2 = fig.add_subplot(gs[0, 1])
	# ax2.plot(iter, [attack_para[1][1]] + mean, color='orange', linewidth=2)
	# ax2.fill_between(iter[1:], lower, upper, facecolor='orange', alpha=0.3)
	# ax2.set_xlabel('Episode', fontsize='15')
	# # ax2.set_ylabel('#Evaded Reviews', fontsize='15')
	# plt.title('YelpNYC', fontsize='20')
	# ax2.set_xticks(np.arange(0, len(iter), 10))
	# ax2.set_yticks(np.arange(1200, attack_para[1][1] + 50, 50))


	# mean = []
	# upper = cp.deepcopy(mean)
	# lower = cp.deepcopy(mean)

	# for a, log in enumerate([len(i) for i in list(remain_reviews_log[6].values())]):
	# 	log1 = [len(i) for i in list(remain_reviews_log[7].values())][a]
	# 	log2 = [len(i) for i in list(remain_reviews_log[8].values())][a]
	# 	ranked = sorted([log, log1, log2])
	# 	lower.append(ranked[0])
	# 	mean.append((ranked[0] + ranked[2])/2.0)
	# 	upper.append(ranked[2])

	# print(mean)

	# ax3 = fig.add_subplot(gs[0, 2])
	# ax3.plot(iter, [attack_para[2][1]] + mean, color='orange', linewidth=2)
	# ax3.fill_between(iter[1:], lower, upper, facecolor='orange', alpha=0.3)
	# ax3.set_xlabel('Episode', fontsize='15')
	# # ax3.set_ylabel('#Evaded Reviews', fontsize='15')
	# plt.title('YelpZip', fontsize='20')
	# ax3.set_xticks(np.arange(0, len(iter), 10))
	# ax3.set_yticks(np.arange(6000, attack_para[2][1] + 50, 500))
	# # plt.subplots_adjust(bottom=0.15, wspace=0.05)
	# plt.show()


if __name__ == '__main__':

	# load metadata and attack defense data
	dataset_name = 'YelpChi'
	prefix = 'Yelp_Dataset/' + dataset_name + '/'
	attack_settings = {'YelpChi': [550, 450, 100], 'YelpNYC': [2200, 1800, 400], 'YelpZip': [9700, 9000, 700]}
	attack_para = attack_settings[dataset_name]
	elite = 10
	top_k = 0.01
	epsilon = 0.1
	lr1 = 30 # learning rate YelpChi: 30, YelpNYC 12, YelpZip 48
	lr2 = 0.01
	iters = 50
	# print('top 15%')
	run_time = 'time1'
	print('dataset {} run time is {}'.format(dataset_name, run_time))
	# # plot the graph
	# with open('Defense/' + 'YelpChi' + '/time1_50_nash_detect.pickle', 'rb') as f:
	# 	attack_log1, detector_log1, loss_log1, pe_log1, remain_reviews_log1 = pickle.load(f)
	# with open('Defense/' + 'YelpChi' + '/time2_50_nash_detect.pickle', 'rb') as f:
	# 	attack_log2, detector_log2, loss_log2, pe_log2, remain_reviews_log2 = pickle.load(f)
	# with open('Defense/' + 'YelpChi' + '/time3_50_nash_detect.pickle', 'rb') as f:
	# 	attack_log3, detector_log3, loss_log3, pe_log3, remain_reviews_log3 = pickle.load(f)
	# with open('Defense/' + 'YelpNYC' + '/time1_50_nash_detect.pickle', 'rb') as f:
	# 	attack_log4, detector_log4, loss_log4, pe_log4, remain_reviews_log4 = pickle.load(f)
	# with open('Defense/' + 'YelpNYC' + '/time2_50_nash_detect.pickle', 'rb') as f:
	# 	attack_log5, detector_log5, loss_log5, pe_log5, remain_reviews_log5 = pickle.load(f)
	# with open('Defense/' + 'YelpNYC' + '/time3_50_nash_detect.pickle', 'rb') as f:
	# 	attack_log6, detector_log6, loss_log6, pe_log6, remain_reviews_log6 = pickle.load(f)
	# with open('Defense/' + 'YelpZip' + '/time1_50_nash_detect.pickle', 'rb') as f:
	# 	attack_log7, detector_log7, loss_log7, pe_log7, remain_reviews_log7 = pickle.load(f)
	# with open('Defense/' + 'YelpZip' + '/time2_50_nash_detect.pickle', 'rb') as f:
	# 	attack_log8, detector_log8, loss_log8, pe_log8, remain_reviews_log8 = pickle.load(f)
	# with open('Defense/' + 'YelpZip' + '/time3_50_nash_detect.pickle', 'rb') as f:
	# 	attack_log9, detector_log9, loss_log9, pe_log9, remain_reviews_log9 = pickle.load(f)
	# plot_result([attack_log1, attack_log2, attack_log3, attack_log4, attack_log5, attack_log6, attack_log7, attack_log8, attack_log9],
	# 			[detector_log1, detector_log2, detector_log3, detector_log4, detector_log5, detector_log6, detector_log7, detector_log8, detector_log9],
	# 			[loss_log1, loss_log2, loss_log3, loss_log4, loss_log5, loss_log6, loss_log7, loss_log8, loss_log9],
	# 			[pe_log1, pe_log2, pe_log3, pe_log4, pe_log5, pe_log6, pe_log7, pe_log8, pe_log9],
	# 			[remain_reviews_log1, remain_reviews_log2, remain_reviews_log3, remain_reviews_log4, remain_reviews_log5, remain_reviews_log6, remain_reviews_log7, remain_reviews_log8, remain_reviews_log9],
	# 			list(attack_settings.values()))
	#
	# exit()

	attacks = ['IncBP', 'IncDS', 'IncPR', 'Random', 'Singleton']
	detectors = ['GANG', 'Prior', 'SpEagle', 'fBox', 'Fraudar']

	setting1 = 'Attack/' + dataset_name + '/LocalBP.pickle'
	setting2 = 'Attack/' + dataset_name + '/IncDS.pickle'
	setting3 = 'Attack/' + dataset_name + '/Random.pickle'
	setting4 = 'Attack/' + dataset_name + '/GlobalPR.pickle'
	setting5 = 'Attack/' + dataset_name + '/Singleton.pickle'

	paths = {'IncBP': setting1, 'IncDS': setting2, 'Random': setting3, 'IncPR': setting4,
			 'Singleton': setting5}

	metadata_filename = prefix + 'metadata.gz'
	user_product_graph, prod_user_graph = read_graph_data(metadata_filename)

	with open(setting1, 'rb') as f:
		evasions = pickle.load(f)
		targets = evasions[1]

	# load fake reviews for each attack to each target item
	item_attack_review = load_fake_reviews(attacks, targets, paths, attack_para)

	with open(paths['IncDS'], 'rb') as f:
		evasions = pickle.load(f)

	# initialize P and Q

	attacks_p = {}
	detectors_q = {}

	# uniformly initialization
	for a in attacks:
		attacks_p[a] = 1 / len(attacks)
	for d in detectors:
		detectors_q[d] = 1 / len(detectors)

	# random initialization
	# for a in attacks:
	# 	attacks_p[a] = np.random.random()
	# for d in detectors:
	# 	detectors_q[d] = np.random.random()
	# detectors_q = {'GANG': 0.4, 'Prior': 0.7, 'SpEagle': 0.9, 'fBox': 0.8, 'Fraudar': 0.1}
	print(detectors_q)


	# initialize loggers
	posted_reviews = {}
	item_attack_mapping = {}
	remain_reviews_log = {}
	attack_log, detector_log = {a: [] for a in attacks_p.keys()}, {d: [] for d in detectors_q.keys()}
	loss_log, pe_log = [], []

	init_time = time.time()

	# start attacks and defenses
	for i in range(0, iters):
		print('Starting episode {}'.format(i))
		posted_reviews[i] = []
		item_attack_mapping[i] = {}

		# generate fake reviews for each item
		for item in targets:
			attack = e_greedy_sample(attacks_p, epsilon)
			item_attack_mapping[i][item] = attack
			reviews = item_attack_review[item][attack]
			posted_reviews[i] += reviews
			# print(attack)

		# run all detectors on all reviews and added reviews
		r_spam_beliefs, r_accu_spam_beliefs, r_detector_belief, top_k_reviews, new_priors, spammer_ids, new_user_product_graph, new_prod_user_graph, added_reviews = run_detectors(user_product_graph, prod_user_graph, posted_reviews[i], detectors_q, top_k)

		# remove the top_k reviews and calculate the practical effect
		print('Compute original Revenue ...')
		ori_RI, ori_ERI, ori_Revenue = compute_re(user_product_graph, prod_user_graph, targets, elite)
		remain_user_product_graph, remain_product_user_graph = remove_topk_reviews(new_user_product_graph, new_prod_user_graph, top_k_reviews)
		print('Compute new Revenue ...')
		new_RI, new_ERI, new_Revenue = compute_re(remain_user_product_graph, remain_product_user_graph, targets, elite)

		# calculate the cost of each posted review
		print('Compute cost ...')
		cost, remain_reviews = compute_cost([ori_RI, ori_ERI, ori_Revenue], [new_RI, new_ERI, new_Revenue], added_reviews, top_k_reviews, targets, new_user_product_graph, elite)

		# compute reward for each item method
		print('Compute reward ...')
		reward = compute_reward([ori_RI, ori_ERI, ori_Revenue], [new_RI, new_ERI, new_Revenue], targets)

		# update sampling probabilities of attacks
		print('Update attack P ...')
		attacks_p = update_p(attacks_p, item_attack_mapping[i], reward, lr2)
		attacks_p = normalize(attacks_p)

		# update weights of detectors
		print('Update defense Q ...')
		detectors_q = update_q(detectors_q, cost, r_detector_belief, r_accu_spam_beliefs, remain_reviews, lr1)
		# detectors_q = normalize(detectors_q)

		# calculate total loss and total practical metric
		total_pe, total_loss = compute_total(ori_Revenue, new_Revenue, targets, cost, r_spam_beliefs)

		print(attacks_p)
		print(detectors_q)
		print('Total PE is {}'.format(total_pe))
		print('Total loss is {}'.format(total_loss))

		# logging
		remain_reviews_log[i] = remain_reviews
		for a, p in attacks_p.items():
			attack_log[a].append(p)
		for d, q in detectors_q.items():
			detector_log[d].append(q)
		loss_log.append(total_loss)
		pe_log.append(total_pe)

		new_time = time.time()
		print('\nTime cost for iteration {} is {}.\n'.format(i, new_time-init_time))
		init_time = new_time

	pickle.dump([attack_log, detector_log, loss_log, pe_log, remain_reviews_log], open('Defense/' + dataset_name + '/' + run_time + '_' + str(iters) + '_' + 'nash_detect.pickle', 'wb'))
	# pickle.dump([attack_log, detector_log, loss_log, pe_log, remain_reviews_log], open('Sensitivity/' + dataset_name + 'top0.15.pickle', 'wb'))
	# plot the result
	# plot_result(attack_log, detector_log, loss_log, pe_log, remain_reviews_log)
