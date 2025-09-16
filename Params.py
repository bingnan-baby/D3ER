import argparse

def ParseArgs():
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--batch', default=1024, type=int, help='batch size')
	parser.add_argument('--tstBat', default=256, type=int, help='number of users in a testing batch')
	parser.add_argument('--stage', default=60, type=int, help='number of stages')
	parser.add_argument('--epoch_per_stage', default=1, type=int, help='number of epochs per stage')
	parser.add_argument('--warmup_epoch', default=20, type=int, help='number of warmup epochs')
	parser.add_argument('--latdim', default=64, type=int, help='embedding size')
	parser.add_argument('--gnn_layer', default=4, type=int, help='number of gnn layers')
	parser.add_argument('--topk', default=20, type=int, help='K of top K')
	parser.add_argument('--data', default='baby', type=str, help='name of dataset')
	parser.add_argument('--reg', default=1e-5, type=float, help='weight decay regularizer')
	parser.add_argument('--temp', default=0.5, type=float, help='temperature in contrastive learning')
	parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
	parser.add_argument('--gpu', default='0', type=str, help='indicates which gpu to use')
	parser.add_argument("--seed", type=int, default=421, help="random seed")
	parser.add_argument("--checkpoints", type=str, default='./checkpoints/baby/', help="save path of checkpoints")
	parser.add_argument("--maxd_v", type=float, default=200, help="distance threshold of visual-HOI and visual-HEI features")
	parser.add_argument("--maxd_t", type=float, default=1.5, help="distance threshold of text-HOI and text-HEI features") #

	parser.add_argument('--keepRate', default=1, type=float, help='ratio of edges to keep')

	# Parameters for boosting
	parser.add_argument('--step_size', default=1, type=float, help='step size of negative gradient in boosting')
	# Parameters for distillation
	parser.add_argument('--lr_distill', default=1e-3, type=float, help='learning rate')
	parser.add_argument('--distill_epoch', default=1, type=int, help='number of epochs per stage')
	parser.add_argument('--distill_reg', default=4, type=float, help='weight for distill loss')
	parser.add_argument('--weight', default=1.0, type=float, help='weight of different ensemble model')

	# Parameters for diffusion of diffmm
	parser.add_argument('--dims', type=str, default='[1000]')
	parser.add_argument('--d_emb_size', type=int, default=10)
	parser.add_argument('--norm', type=bool, default=False)
	parser.add_argument('--steps', type=int, default=5)
	parser.add_argument('--noise_scale', type=float, default=0.1)
	parser.add_argument('--noise_min', type=float, default=0.0001)
	parser.add_argument('--noise_max', type=float, default=0.02)
	parser.add_argument('--sampling_noise', type=bool, default=False)
	parser.add_argument('--sampling_steps', type=int, default=0)

	parser.add_argument('--rebuild_k', type=int, default=1)
	parser.add_argument('--e_loss', type=float, default=0.01)
	parser.add_argument('--ris_lambda', type=float, default=0.5)
	parser.add_argument('--ris_adj_lambda', type=float, default=0.2)
	parser.add_argument('--trans', type=int, default=0, help='0: R*R, 1: Linear, 2: allrecipes')
	parser.add_argument('--cl_method', type=int, default=0, help='0:m vs m ; 1:m vs main')


	parser.add_argument('--sp_reg', default=1e-2, type=float, help='weight for specific loss')
	parser.add_argument('--ssl_reg', default=1e-1, type=float, help='weight for contrative learning')
	# swd
	parser.add_argument('--sw_reg', default=0.0001, type=float, help='weight of wasserstein distance')
	parser.add_argument('--n_projection', default=128, type=int, help='numbers of projectors')
	return parser.parse_args()
args = ParseArgs()
