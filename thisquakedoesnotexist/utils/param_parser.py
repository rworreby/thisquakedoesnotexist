#!/usr/bin/env python3

import argparse


class ParamParser:
    def __init__(self):
        self.args = None

    @staticmethod
    def parse_args(args):
        parser = argparse.ArgumentParser()

        # Setup arguments
        setup_args = parser.add_argument_group('Setup Parameters')
        setup_args.add_argument('--gpus', type=int, default=1,
                                help="Device to use. Set to 1 for GPU (default), 0 for CPU.")
        setup_args.add_argument('--name', type=str, default='Unnamed', help='Set the name of the Model')
        setup_args.add_argument('--data_file', type=str, default='thisquakedoesnotexist/data/japan/waveforms.npy',
                                help='Path to waveforms data file. Needs to by in .npy format.\
                                    Default: "thisquakedoesnotexist/data/japan/waveforms.npy"')
        setup_args.add_argument('--attr_file', type=str, default='thisquakedoesnotexist/data/japan/attributes.csv',
                                help='Path to attributes data file. Needs to by in .csv format.\
                                    Default: thisquakedoesnotexist/data/japan/attributes.csv"')
        setup_args.add_argument('--model_file', type=str, default='thisquakedoesnotexist/models/gan_d1.py',
                                help='Path to GAN model architecture file.\
                                    Default: "thisquakedoesnotexist/models/gan_d1.py"')    

        # Model parameters
        model_args = parser.add_argument_group('Model Parameters')
        model_args.add_argument('--lt', type=int, default=1000, 
                                help='Size of input to discriminator. Default: 1000')
        model_args.add_argument('--dt', type=float, default=0.05, 
                                help='Time step size in data. Ddfault: 0.05 (20 Hz)')
        model_args.add_argument('--noise_dim', type=int, default=100, 
                                help='Dimension of Gaussian noise. Default: 100')
        model_args.add_argument('--lr', type=float, default=1e-4, 
                                help='Learning rate of optimizer. Default: 1e-4')
        model_args.add_argument('--n_critic', type=int, default=10,
                                help='Number of critic iterations per GAN training cycle. Default: 10')
        model_args.add_argument('--gp_lambda', type=float, default=10.0,
                                help='?. Default: 10')
        model_args.add_argument('--beta1', type=float, default=0.0,
                                help='Adam optimizer parameter beta1. Default: 0.0')
        model_args.add_argument('--beta2', type=float, default=0.9,
                                help='Adam optimizer parameter beta2. Default: 0.9')
        model_args.add_argument('--sample_rate', type=int, default=100,
                                help='Sample rate of data in Hz. Default: 100')
        model_args.add_argument('--n_cond_bins', type=int, default=20,
                                help='Number of bins for conditional variables distance, magnitude, and vs30. Default: 20')
  
        
        # Training Parameters
        training_args = parser.add_argument_group('Training Parameters')
        training_args.add_argument('--epochs', type=int, default=10,
                                   help="Set the number of epochs to train. Default: 10")
        training_args.add_argument('--batch_size', type=int, default=32,
                                   help="Set the batch size for training. Default: 32")
        training_args.add_argument('--frac_train', type=float, default=0.8,
                                   help='Fraction of data used for training (rest is used for validation). \
                                   Default: 0.8')    
        
        # Output Parameters
        output_args = parser.add_argument_group('Output Parameters')
        output_args.add_argument('--output_dir', type=str, default='thisquakedoesnotexist/data/output', 
                                help='Path of output directory. Default: "thisquakedoesnotexist/data/output/')
        output_args.add_argument('--print_freq', type=int, default=400, 
                                help='Frequency for printing the parameter set (every x iterations). Default: 400')
        output_args.add_argument('--save_freq', type=int, default=1,
                                help='Frequency for saving the model (every x iterations). Default: 1')
        output_args.add_argument('--n_pltots', type=int, default=20,
                                 help='Number of synthetic plots to generate. Default: 20')
        
        # MLFLow Parameters 
        output_args = parser.add_argument_group('MLFlow Parameters')
        output_args.add_argument('--tracking_uri', type=str, default='/home/rworreby/thisquakedoesnotexist/mlruns/', 
                                help='Path of MLFLow tracking URI. Default: "/home/rworreby/thisquakedoesnotexist/mlruns/"')
        output_args.add_argument('--experiment_name', type=str, default='Unnamed_Experiment', 
                                help='Name of MLFlow experiment. Default: Unnamed_Experiment')
        
        return parser.parse_args(args)