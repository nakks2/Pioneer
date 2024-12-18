from mpmath import eigh
import torch.nn.functional as F

from lib.likelihood_eval import *
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import torch.nn as nn
import torch
import lib.utils as utils
import numpy as np
import geoopt
from scipy.linalg import eigh
import torch.nn as nn
mse_loss = nn.MSELoss(reduction='sum')

class VAE_Baseline(nn.Module):
	def __init__(self,device,feature_max,feature_min):

		super(VAE_Baseline, self).__init__()
		

		self.device = device
		obsrv_std = 0.01
		obsrv_std = torch.Tensor([obsrv_std]).to(device)
		z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
		self.obsrv_std = torch.Tensor([obsrv_std]).to(device)

		self.z0_prior = z0_prior
		self.feature_max = feature_max
		self.feature_min = feature_min

	def compute_all_losses(self, batch_dict_encoder,batch_dict_decoder,batch_dict_graph ,num_atoms,edge_lamda, kl_coef = 1.,istest=False,ce=False,dataset=None):
		'''

		:param batch_dict_encoder:
		:param batch_dict_decoder: dict: 1. time 2. data: (K*N, T2, D)
		:param batch_dict_graph: #[K,T2,N,N], ground_truth graph with log normalization
		:param num_atoms:
		:param kl_coef:
		:return:
		'''

		pred_node,pred_edge, info= self.get_reconstruction(batch_dict_encoder,batch_dict_decoder)
		# pred_node [ K*N , time_length, d]
		# pred_edge [ K*N*N, time_length, d]
		pred_edge = pred_edge.view(pred_edge.shape[0], pred_edge.shape[1],1)
		if istest:
			mask_index = batch_dict_decoder["masks"]
			pred_node = pred_node[:,mask_index,:]
			pred_edge = pred_edge[:,mask_index,:]



		# Reshape batch_dict_graph
		k = batch_dict_graph.shape[0]
		T2 = batch_dict_graph.shape[1]
		truth_graph = torch.reshape(batch_dict_graph,(k,T2,-1)).float() # [K,T,N*N]
		truth_graph = torch.unsqueeze(truth_graph.permute(0,2,1),dim=3) #[K,N*N,T,1]
		truth_graph = torch.reshape(truth_graph,(-1,T2,1)) #[K*N*N,T,1]

		fp_mu, fp_std, fp_enc = info["first_point"]  # [K*N,D]
		if dataset =="social":
			fp_std = fp_std.abs()

			fp_distr = Normal(fp_mu, fp_std)

			assert (torch.sum(fp_std < 0) == 0.)
			kldiv_z0 = kl_divergence(fp_distr, self.z0_prior)  # [K*N,D_ode_latent]

			# Mean over number of latent dimensions
			kldiv_z0 = torch.mean(kldiv_z0)  # Contains infinity.

			# Compute likelihood of all the points
			rec_likelihood_node = self.get_gaussian_likelihood(
				batch_dict_decoder["data"], pred_node,
				mask=None)  # negative value

			rec_likelihood_edge = self.get_gaussian_likelihood(
				truth_graph, pred_edge,
				mask=None)  # negative value

			rec_likelihood = rec_likelihood_node + edge_lamda * rec_likelihood_edge

			mape_node = self.get_loss(
				batch_dict_decoder["data"], pred_node, truth_gt=batch_dict_decoder["data_gt"],
				mask=None, method='MAPE', istest=istest)  # [1]

			mse_node = self.get_loss(
				batch_dict_decoder["data"], pred_node,
				mask=None, method='MSE', istest=istest)  # [1]

			# loss

			loss = - torch.logsumexp(rec_likelihood - kl_coef * kldiv_z0, 0)
			if torch.isnan(loss):
				loss = - torch.mean(rec_likelihood - kl_coef * kldiv_z0, 0)
		else :
			loss_node = mse_loss(batch_dict_decoder["data"], pred_node)
			loss_edge = mse_loss(truth_graph, pred_edge)
			loss = loss_node + edge_lamda * loss_edge
			if dataset=="CReSIS": #no zeors
				# mask = batch_dict_decoder["data"] != 0
				y_true = batch_dict_decoder["data"]*(self.feature_max-self.feature_min)+self.feature_min
				y_pred = pred_node*(self.feature_max-self.feature_min)+self.feature_min
				absolute_percentage_error = torch.abs((y_true - y_pred) / y_true)
				y_true = y_true.view(-1)
				y_pred = y_pred.view(-1)
			elif dataset=="weather": #has zeros
				mask = batch_dict_decoder["data"] != 0
				y_true = batch_dict_decoder["data"][mask]*(self.feature_max-self.feature_min)
				y_pred = pred_node[mask]*(self.feature_max-self.feature_min)
				absolute_percentage_error = torch.abs((y_true - y_pred) / y_true)

			mape_node = torch.mean(absolute_percentage_error)

			mse_node = mse_loss(y_true, y_pred)/y_true.shape[0]

		results = {}
		results["loss"] = loss
		results["MAPE"] = torch.mean(mape_node).data.item()
		results["MSE"] = torch.mean(mse_node).data.item()
		if ce:
			entropy = []
			pred_edge = pred_edge.view(pred_edge.shape[0], pred_edge.shape[1])  #[K*N*N,T]
			for t in range(pred_edge.shape[1]):
				edge_t = pred_edge[:,t]
				e = self.compute_von_neumann_entropy_from_coo(edge_t)
				entropy.append(e)
			results["entropy"] = entropy
		return results

	def compute_von_neumann_entropy_from_coo(self,A):

		# Compute degree matrix D
		A = A.detach().cpu().numpy()
		A = A.reshape(80,80)
		D = np.diag(A.sum(0))

		epsilon = 1e-5
		# Compute normalized Laplacian L
		D_norm = np.diag(np.power(A.sum(0), -1 / 2))
		L = D - A
		L_norm = np.matmul(np.matmul(D_norm, L), D_norm)


		# Compute eigenvalues of L
		eigenvalues = eigh(L_norm, eigvals_only=True)
		mask = eigenvalues <= 0
		eigenvalues[mask] = epsilon
		# Compute von Neumann entropy Ht
		entropy = -np.sum(eigenvalues * np.log(eigenvalues))

		return entropy
	def get_gaussian_likelihood(self, truth, pred_y,temporal_weights=None, mask=None ):
		# pred_y shape [K*N, n_tp, n_dim]
		# truth shape  [K*N, n_tp, n_dim]

		# Compute likelihood of the data under the predictions

		log_density_data = masked_gaussian_log_density(pred_y, truth,
			obsrv_std = self.obsrv_std, mask = mask,temporal_weights= temporal_weights) #【num_traj = K*N] [250,3]
		log_density = torch.mean(log_density_data)

		# shape: [n_traj_samples]
		return log_density

	def get_loss(self, truth, pred_y, truth_gt=None,mask = None,method='MSE',istest=False):
		# pred_y shape [n_traj, n_tp, n_dim]
		# truth shape  [n_traj, n_tp, n_dim]

		#Transfer from inc to cum

		truth = utils.inc_to_cum(truth)
		pred_y = utils.inc_to_cum(pred_y)
		num_times = truth.shape[1]
		time_index = [num_times-1] # last timestamp

		if istest:
			truth = truth[:,time_index,:]
			pred_y = pred_y[:,time_index,:]   #[N,1,D]
			if truth_gt != None:
				truth_gt = truth_gt[:,time_index,:]

		# Compute likelihood of the data under the predictions
		log_density_data = compute_loss(pred_y, truth, truth_gt,mask = mask,method=method)
		# shape: [1]
		return torch.mean(log_density_data)

