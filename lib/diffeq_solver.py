import torch
import torch.nn as nn
from geoopt import StereographicExact
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
import lib.utils as utils
import torch.nn.functional as F
from scipy.linalg import block_diag
from torch_scatter import scatter_add
import scipy.sparse as sp
from geoopt.manifolds.stereographic.math import expmap, project, dist
from torch.func import vjp, vmap

def compute_edge_initials(first_point_enc, num_atoms,encoder_edge):
    '''

    :param first_point_enc: [K*N,D]
    :return: [K*N*N,D]
    '''
    node_feature_num = first_point_enc.shape[1]
    fully_connected = np.ones([num_atoms, num_atoms])
    rel_send = np.array(utils.encode_onehot(np.where(fully_connected)[0]),
                        dtype=np.float32)  # every node as one-hot[10000], (N*N,N)
    rel_rec = np.array(utils.encode_onehot(np.where(fully_connected)[1]),
                       dtype=np.float32)  # every node as one-hot[10000], (N*N,N)

    rel_send = torch.FloatTensor(rel_send).to(first_point_enc.device)
    rel_rec = torch.FloatTensor(rel_rec).to(first_point_enc.device)

    first_point_enc = first_point_enc.view(-1, num_atoms, node_feature_num)  # [K,N,D]

    senders = torch.matmul(rel_send, first_point_enc)  # [K,N*N,D]
    receivers = torch.matmul(rel_rec, first_point_enc)  # [K,N*N,D]

    edge_initials = torch.cat([senders, receivers], dim=-1)  # [K,N*N,2D]
    edge_initials = F.gelu(encoder_edge(edge_initials))  # [K,N*N,D_edge]
    edge_initials = edge_initials.view(-1, edge_initials.shape[2])  # [K*N*N,D_edge]

    return edge_initials

class DiffeqSolver(nn.Module):
    def __init__(self, ode_func,args,
            odeint_rtol = 1e-3, odeint_atol = 1e-4, device = torch.device("cpu")):
        super(DiffeqSolver, self).__init__()

        self.device = device
        self.ode_func = ode_func
        self.args = args
        self.num_atoms = args.num_atoms

        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol
        self.manifold = StereographicExact(k=1, learnable=False)
        if args.Riemannan == True:
            self.kappa=1
        else:
            self.kappa=0

    def forward(self, first_point,time_steps_to_predict,encoder_edge):
        '''

        :param first_point:  [K*N,D]
        :param edge_initials: [K*N*N,D]
        :param time_steps_to_predict: [t]
        :return:
        '''

        # Node ODE Function
        n_traj,feature_node = first_point.size()[0], first_point.size()[1]  #[K*N,d]
        if self.args.augment_dim > 0:
            aug_node = torch.zeros(first_point.shape[0], self.args.augment_dim).to(self.device) #[K*N,D_aug]
            first_point = torch.cat([first_point, aug_node], 1) #[K*N,d+D_aug]
            feature_node += self.args.augment_dim

        # Edge initialization: h_ij = f([u_i,u_j])
        edge_initials = compute_edge_initials(first_point, self.num_atoms, encoder_edge)  # [K*N*N,D]
        assert (not torch.isnan(edge_initials).any())

        node_edge_initial = torch.cat([first_point,edge_initials],0)  #[K*N + K*N*N,D+D_aug]
        # Set index
        K_N = int(node_edge_initial.shape[0]/(self.num_atoms+1))
        K = K_N/self.num_atoms
        self.ode_func.set_index_and_graph(K_N,K)

        node_initial = node_edge_initial[:K_N,:]
        self.ode_func.set_initial_z0(node_initial)

        # Results
        pred_y,v = self.integrator(self.ode_func, node_edge_initial, time_steps_to_predict,self.kappa,first_point.shape[0]) #[time_length, K*N + K*N*N, D]

        pred_y = pred_y.permute(1,0,2) #[ K*N + K*N*N, time_length, d]

        assert(pred_y.size()[0] == K_N*(self.num_atoms+1))

        if self.args.augment_dim > 0:
            pred_y = pred_y[:, :, :-self.args.augment_dim]

        return pred_y,K_N
    def integrator(self,ode_func,x0,t,kappa,split_length):
        t0s = t[:-1]
        t1s = t[1:]
        vts = []
        xts = [x0]

        xt = x0
        for t0, t1 in zip(t0s, t1s):
            dt = t1 - t0
            vt = ode_func(t, xt)
            xt_node = xt[:split_length,:]
            xt_edge = xt[split_length:,:]
            vt_node = vt[:split_length,:]
            vt_edge = vt[split_length:,:]
            xt_node = self.ode_step(xt_node, vt_node, dt, self.kappa)
            xt_edge = self.ode_step(xt_edge, vt_edge, dt, 0)
            if kappa != 0:
                xt_node = project(xt_node, k=self.manifold.k)
            xt = torch.cat([xt_node, xt_edge])
            vts.append(vt)
            xts.append(xt)
        # vts.append(ode_func(t1，xt))
        # return torch.stack(xts),torch.stack(vts)
        return torch.stack(xts), None

    def ode_step(self,xt, vt, dt, kappa):
        if kappa != 0:
            value = expmap(xt,vt*dt, k=self.manifold.k)
            return project(value,k=self.manifold.k)
        else:
            return xt+vt*dt
class ODEFunc(nn.Module):
    def __init__(self, node_ode_func_net,edge_ode_func_net,num_atom, dropout,device = torch.device("cpu")):
        """
        input_dim: dimensionality of the input
        latent_dim: dimensionality used for ODE. Analog of a continous latent state
        """
        super(ODEFunc, self).__init__()

        self.device = device
        self.node_ode_func_net = node_ode_func_net  #input: x, edge_index
        self.edge_ode_func_net = edge_ode_func_net
        self.num_atom = num_atom
        self.nfe = 0
        self.dropout = nn.Dropout(dropout)


    def forward(self, t_local, z, backwards = False):
        """
        Perform one step in solving ODE. Given current data point y and current time point t_local, returns gradient dy/dt at this time point

        t_local: current time point
        z:  [H,E] concat by axis0. H is [K*N,D], E is[K*N*N,D], z is [K*N + K*N*N, D]
        """
        self.nfe += 1


        node_attributes = z[:self.K_N,:]
        edge_attributes = z[self.K_N:,:]
        assert (not torch.isnan(node_attributes).any())
        assert (not torch.isnan(edge_attributes).any())

        #grad_edge, edge_value = self.edge_ode_func_net(node_attributes,self.num_atom) # [K*N*N,D],[K,N*N], edge value are non-negative by using relu.
        grad_edge, edge_value = self.edge_ode_func_net(node_attributes, edge_attributes,
                                                       self.num_atom,)  # [K*N*N,D],[K,N*N], edge value are non-negative by using relu.todo:with self-evolution
        edge_value = self.normalize_graph(edge_value, self.K_N)
        assert (not torch.isnan(edge_value).any())
        grad_node = self.node_ode_func_net(node_attributes,edge_value,self.node_z0) # [K*N,D]
        assert (not torch.isnan(grad_node).any())
        assert (not torch.isinf(grad_edge).any())

        assert (not torch.isnan(grad_node).any())
        assert (not torch.isinf(grad_edge).any())


        # Concat two grad
        grad = self.dropout(torch.cat([grad_node, grad_edge], 0))  # [K*N + K*N*N, D]

        return grad


    def set_index_and_graph(self,K_N,K):
        '''

        :param K_N: index for separating node and edge matrixs.
        :return:
        '''
        self.K_N = K_N
        self.K = int(K)
        self.K_N_N = self.K_N*self.num_atom
        self.nfe = 0

        # Step1: Concat into big graph: which is set by default.diagonal matrix
        edge_each = np.ones((self.num_atom, self.num_atom))
        edge_whole = block_diag(*([edge_each] * self.K))
        edge_index, _ = utils.convert_sparse(edge_whole)
        self.edge_index = torch.LongTensor(edge_index).to(self.device)

    def set_initial_z0(self,node_z0):
        self.node_z0 = node_z0

    def normalize_graph(self,edge_weight,num_nodes):
      '''
      For asymmetric graph
      :param edge_index: [num_edges,2], torch.LongTensor
      :param edge_weight: [K,N*N] ->[num_edges]
      :param num_nodes:
      :return:
      '''
      assert (not torch.isnan(edge_weight).any())
      assert (torch.sum(edge_weight<0)==0)
      edge_weight_flatten = edge_weight.view(-1)  #[K*N*N]

      row, col = self.edge_index[0], self.edge_index[1]
      deg = scatter_add(edge_weight_flatten, row, dim=0, dim_size=num_nodes) #[K*N]
      deg_inv_sqrt = deg.pow_(-1)
      deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
      if torch.isnan(deg_inv_sqrt).any():
          assert (torch.sum(deg == 0) == 0)
          assert (torch.sum(deg < 0) == 0)

      assert (not torch.isnan(deg_inv_sqrt).any())


      edge_weight_normalized = deg_inv_sqrt[row] * edge_weight_flatten   #[k*N*N]
      assert (torch.sum(edge_weight_normalized < 0) == 0) and (torch.sum(edge_weight_normalized > 1) == 0)

      # Reshape back

      edge_weight_normalized = torch.reshape(edge_weight_normalized,(self.K,-1)) #[K,N*N]
      assert (not torch.isnan(edge_weight_normalized).any())

      return edge_weight_normalized











