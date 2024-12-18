from lib.gnn_models import GNN,Node_GCN,Ricci,Decoder
from lib.latent_ode import ODE
from lib.diffeq_solver import DiffeqSolver,ODEFunc
from lib.utils import print_parameters
import torch.nn as nn
import lib.utils as utils


def create_ODE_model(args, input_dim, device):


	# dim related
	input_dim = input_dim
	output_dim = args.output_dim
	ode_hidden_dim = args.ode_dims
	rec_hidden_dim = args.rec_dims


	#ODE related
	if args.augment_dim > 0:  # Padding is done after the output of encoder. Encoder output dim is the expected ode_hidden_dim. True hidden dim is ode_hidden_dim + augment_dim
		ode_input_dim = ode_hidden_dim + args.augment_dim

	else:
		ode_input_dim = ode_hidden_dim


	# rec_ouput_dim = ode_hidden_dim*2 # Need to split the vector into mean and variance (multiply by 2)


	#Encoder related

	encoder_z0 = GNN(in_dim=input_dim, n_hid=rec_hidden_dim, out_dim=ode_hidden_dim, n_heads=1,
						 n_layers=args.rec_layers, dropout=args.dropout, conv_name=args.z0_encoder,is_encoder=True, args = args)  # [b,n_ball,e]
	# ODE related
	# 1. Node ODE function
	node_ode_func_net = Node_GCN(in_dims = ode_input_dim,out_dims = ode_input_dim,num_atoms = args.num_atoms,dropout=args.dropout,Rie=args.Riemannan)

	# 2. Edge ODE function
	w_node_to_edge_initial = nn.Linear(ode_input_dim * 2, ode_input_dim)  # h_ij = W([h_i||h_j])
	utils.init_network_weights(w_node_to_edge_initial)
	w_node_to_edge_initial = w_node_to_edge_initial.to(device)
	edge_ode_func_net = Ricci(feature_dim=ode_input_dim,w_node2edge = w_node_to_edge_initial,alpha=args.alpha,beta = args.beta,dropout=args.dropout,num_atoms=args.num_atoms,device=device)

	# 3. Wrap Up ODE Function
	coupled_ode_func = ODEFunc(
		node_ode_func_net=node_ode_func_net,
		edge_ode_func_net=edge_ode_func_net,
		device=device,
		num_atom = args.num_atoms,dropout=args.dropout).to(device)



	diffeq_solver = DiffeqSolver(coupled_ode_func, args=args,odeint_rtol=1e-2, odeint_atol=1e-2, device=device)

    #Decoder related
	decoder_node = Decoder(ode_hidden_dim, output_dim).to(device)
	decoder_edge = Decoder(ode_hidden_dim,1).to(device)

	model = ODE(
		w_node_to_edge_initial=w_node_to_edge_initial,
		ode_hidden_dim = ode_hidden_dim,
		encoder_z0 = encoder_z0,
		decoder_node = decoder_node,
		decoder_edge = decoder_edge,
		diffeq_solver = diffeq_solver,
		device = device,
		Rie = args.Riemannan,
		feature_max = args.feature_max,
		feature_min = args.feature_min,
		).to(device)

	print_parameters(model)


	return model
