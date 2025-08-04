import torch

def get_hyperparameters():
    return {
        'lr': 1e-5,
        'decay': 0.0, 
        'smooth': 1e-6,
        'dropout': 0.0, 
        'lr_decay_factor': 0.95,
        'patience' : 2,
        'dataset_name': 'GAT_6_501_chm13HIFI',
        'wandb_mode': 'disabled',  # switch between 'online' and 'disabled'
        'test_mode': False,  # switch between True and False
        'num_epochs': 250,
        'hidden_features': 32,
        'hidden_edge_scores': 32,
        'nb_pos_enc': 16,
        'num_layers': 6,
        'edge_features': 2,
        'node_features': 1,
        'num_parts_metis_train': 250,#250
        'num_parts_metis_eval': 250,#250
        'batch_size_train': 10,#25 10
        'batch_size_eval': 10,
        'num_workers': 3,
        'seed': 0,
        'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
        'batch_norm': True,
        'nheads': 8,
        'alpha_gat': 2e-1,
        'best_loss': float('inf'),

        'save_train_graph_path': 'dataset/train',
        'save_val_graph_path': 'dataset/val',
        'save_test_graph_path': 'dataset/test',
        'save_garph_path': '/home/xl/Paper/KANGNN_for_Assembly/checkpoints/graph',
    }

