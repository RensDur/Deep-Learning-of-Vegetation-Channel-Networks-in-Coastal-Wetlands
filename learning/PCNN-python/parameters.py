import argparse


def str2bool(v):
    """
    'boolean type variable' for add_argument
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('boolean value expected.')


def params():
    """
    return parameters for training / testing / plotting of models
    :return: parameter-Namespace
    """
    parser = argparse.ArgumentParser(description='train / test a pytorch model to predict frames')

    # Training parameters
    parser.add_argument('--net', default="UNetSWE", type=str, help='network to train (default: UNet2)',
                        choices=["UNet1", "UNet2", "UNet3", "UNetSWE"])
    parser.add_argument('--n_epochs', default=1000, type=int,
                        help='number of epochs (after each epoch, the model gets saved)')
    parser.add_argument('--n_grad_steps', default=500, type=int, help='number of gradient descent steps')
    parser.add_argument('--hidden_size', default=64, type=int, help='hidden size of network (default: 20)')
    parser.add_argument('--n_batches_per_epoch', default=5000, type=int,
                        help='number of batches per epoch (default: 5000)')
    parser.add_argument('--batch_size', default=100, type=int, help='batch size (default: 100)')
    parser.add_argument('--n_time_steps', default=1, type=int,
                        help='number of time steps to propagate gradients (default: 1)')  # note: this only works with static environments (and didn't bring any benefits anyway)
    parser.add_argument('--average_sequence_length', default=5000, type=int,
                        help='average sequence length in dataset (default: 5000)')
    parser.add_argument('--dataset_size', default=1000, type=int, help='size of dataset (default: 1000)')
    parser.add_argument('--cuda', default=True, type=str2bool, help='use GPU')

    parser.add_argument('--loss_h', default=1, type=float, help='Weight of loss factor theta_0')
    parser.add_argument('--loss_momentum', default=1, type=float, help='Weight of loss factor theta_1')
    parser.add_argument('--loss_S', default=0, type=float, help='Weight of loss factor theta_2')
    parser.add_argument('--loss_B', default=0, type=float, help='Weight of loss factor theta_3')
    parser.add_argument('--loss_bound', default=1e6, type=float, help='loss factor for boundary conditions')
    parser.add_argument('--loss_reg', default=1, type=float, help='Weight of regularizers in loss')
    parser.add_argument('--plot_loss', default=False, type=str2bool, help='Plot loss-image alongside losses over time')

    parser.add_argument('--regularize_grad_p', default=0, type=float,
                        help='regularizer for gradient of p. evt needed for very high reynolds numbers (default: 0)')
    parser.add_argument('--max_speed', default=1, type=float,
                        help='max speed for boundary conditions in dataset (default: 1)')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate of optimizer (default: 0.001)')
    parser.add_argument('--lr_grad', default=0.001, type=float, help='learning rate of optimizer (default: 0.001)')
    parser.add_argument('--clip_grad_norm', default=None, type=float, help='gradient norm clipping (default: None)')
    parser.add_argument('--clip_grad_value', default=None, type=float, help='gradient value clipping (default: None)')
    parser.add_argument('--log', default=True, type=str2bool,
                        help='log models / metrics during training (turn off for debugging)')
    parser.add_argument('--log_grad', default=False, type=str2bool,
                        help='log gradients during training (turn on for debugging)')
    parser.add_argument('--plot_sqrt', default=False, type=str2bool,
                        help='plot sqrt of velocity value (to better distinguish directions at low velocities)')
    parser.add_argument('--plot', default=False, type=str2bool, help='plot during training')
    parser.add_argument('--flip', default=False, type=str2bool,
                        help='flip training samples randomly during training (default: False)')
    parser.add_argument('--integrator', default='imex', type=str,
                        help='integration scheme (explicit / implicit / imex) (default: imex)',
                        choices=['explicit', 'implicit', 'imex'])
    parser.add_argument('--loss', default='square', type=str, help='loss type to train network (default: square)',
                        choices=['square'])
    parser.add_argument('--loss_multiplier', default=1, type=float, help='multiply loss / gradients (default: 1)')
    parser.add_argument('--target_freq', default=7, type=float,
                        help='target frequency of optimal control algorithm (default: 7; choose value between 2-8)')

    # Setup parameters
    parser.add_argument('--width', default=200, type=int, help='setup width')
    parser.add_argument('--height', default=200, type=int, help='setup height')
    parser.add_argument('--separation', default=0.01, type=float, help='cell separation in meters')

    # Domain parameters
    parser.add_argument('--Hin', default=1e-5, type=float, help="")
    parser.add_argument('--Hc', default=1e-3, type=float, help="")
    parser.add_argument('--H0', default=0.02, type=float, help="Initial water thickness")
    parser.add_argument('--grav', default=9.81, type=float, help="")
    parser.add_argument('--rho', default=1000, type=float, help="Water density")
    parser.add_argument('--Du', default=0.5, type=float, help="Turbulent Eddy velocity")
    parser.add_argument('--nb', default=0.016, type=float, help="bed roughness for bare land")
    parser.add_argument('--nv', default=0.2, type=float, help="bed roughness for vegetated land")
    parser.add_argument('--k', default=1500, type=float, help="Vegetation carrying capacity")
    parser.add_argument('--D0', default=1e-7, type=float, help="Sediment diffusivity in absence of vegetation")
    parser.add_argument('--pD', default=0.99, type=float, help="fraction by which sediment diffusivity is reduced when vegetation is at carrying capacity")
    parser.add_argument('--Sin', default=5e-9, type=float, help="Maximum sediment input rate")
    parser.add_argument('--Qs', default=6e-4, type=float, help="water layer thickness at which sediment input is halved")
    parser.add_argument('--Es', default=2.5e-4, type=float, help="Sediment erosion rate")
    parser.add_argument('--pE', default=0.9, type=float, help="Fraction by which sediment erosion is reduced when vegetation is at carrying capacity")
    parser.add_argument('--r', default=3.2e-8, type=float, help="Intrinsic plant growth rate (=1 per year)")
    parser.add_argument('--Qq', default=0.02, type=float, help="Water layer thickness at which vegetation growth is halved")
    parser.add_argument('--EB', default=1e-5, type=float, help="Vegetation erosion rate")
    parser.add_argument('--DB', default=6e-9, type=float, help="Vegetation diffusivity")
    parser.add_argument('--morphological_acc_factor', default=44712, type=float, help="Morphological acceleration factor, required for S and B")
    parser.add_argument('--pEst', default=0.002, type=float, help="Probability of vegetation seedling establishment")
    parser.add_argument('--dt', default=0.0001, type=float, help='timestep of fluid integrator')

    # Load parameters
    parser.add_argument('--load_date_time', default=None, type=str, help='date_time of run to load (default: None)')
    parser.add_argument('--load_index', default=None, type=int, help='index of run to load (default: None)')
    parser.add_argument('--load_optimizer', default=False, type=str2bool,
                        help='load state of optimizer (default: True)')
    parser.add_argument('--load_latest', default=False, type=str2bool,
                        help='load latest version for training (if True: leave load_date_time and load_index None. default: False)')

    # parse parameters
    params = parser.parse_args()

    return params


def get_description(params):
    return f"net {params.net}; hs {params.hidden_size}; dt {params.dt};"
