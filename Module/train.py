import os
import json
import numpy as np
import torch
import torch.autograd.profiler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import dgl
import glob
import os
import wandb
import globals
from datetime import datetime
from colorama import Fore, Style
from hyperparameters import get_hyperparameters
import models
import utils
from utils import add_positional_encoding

wandb.require("core")

def view_model_param(model):
    """Get the total number of parameters of the model.
    
    Parameters
    ----------
    model : torch.nn.Module
        PyTorch model for which the number of parameters is calculated

    Returns
    -------
    int
        Number of parameters of the model
    """
    total_param = 0
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    return total_param

def generate_adj_matrix(graph, device):
    adj_matrix = graph.adjacency_matrix(scipy_fmt='coo').tocoo()  # 转换为 COO 格式
    # 获取行和列的索引以及对应的值
    row, col = adj_matrix.nonzero()
    indices = torch.LongTensor(np.stack((row, col), axis=0)).to(device)  # 合并并移动到指定设备
    values = torch.FloatTensor(adj_matrix.data).to(device)  # 移动到指定设备
    shape = torch.Size(adj_matrix.shape)
    adj_torch = torch.sparse_coo_tensor(indices, values, shape).to(device)  # 移动到指定设备
    return adj_torch

def gen_train_batch(dgl_graph_list):
    if globals.batch_size_train <= 1:
        return dgl_graph_list
    else:
        dataloaders = list()
        for id, graph in enumerate(dgl_graph_list):
            try:
                os.remove(globals.cluster_cache_path)
            except:
                pass 
            # Run Metis
            print(f"Start conver No.{id+1} train_graph format...")
            g = graph.long()
            num_clusters = torch.LongTensor(1).random_(globals.num_parts_metis_train-100,globals.num_parts_metis_train+100).item() # DEBUG!!!
            sampler = dgl.dataloading.ClusterGCNSampler(g, num_clusters, cache_path=globals.cluster_cache_path) 
            dataloader = dgl.dataloading.DataLoader(g, torch.arange(num_clusters), sampler, batch_size=globals.batch_size_train, shuffle=True, drop_last=False, num_workers=globals.num_workers)
            dataloaders.append(dataloader)
        return dataloaders

def gen_val_test_batch(dgl_graph_list):
    if globals.batch_size_train <= 1:
        return dgl_graph_list
    else:
        dataloaders = list()
        for id, graph in enumerate(dgl_graph_list):
            try:
                os.remove(globals.cluster_cache_path)
            except:
                pass 
            # Run Metis
            print(f"Start conver No.{id+1} val_test_graph format...")
            g = graph.long()
            sampler = dgl.dataloading.ClusterGCNSampler(g, globals.num_parts_metis_eval, cache_path=globals.cluster_cache_path) 
            dataloader = dgl.dataloading.DataLoader(g, torch.arange(globals.num_parts_metis_eval), sampler, batch_size=globals.batch_size_eval, shuffle=True, drop_last=False, num_workers=globals.num_workers)
            dataloaders.append(dataloader)
        return dataloaders

def train(out='glimmer'):
    hyperparameters = get_hyperparameters()
    seed = hyperparameters['seed']
    lr = hyperparameters['lr']
    num_epochs = hyperparameters['num_epochs']
    hidden_features = hyperparameters['hidden_features']
    node_features = hyperparameters['node_features']
    edge_features = hyperparameters['edge_features']
    dataset_name = hyperparameters['dataset_name']
    wandb_mode = hyperparameters['wandb_mode']
    hidden_edge_scores = hyperparameters['hidden_edge_scores']
    num_layers = hyperparameters['num_layers']
    globals.nb_pos_enc = hyperparameters['nb_pos_enc']
    globals.num_parts_metis_train = hyperparameters['num_parts_metis_train']
    globals.num_parts_metis_eval = hyperparameters['num_parts_metis_eval']
    globals.batch_size_train = hyperparameters['batch_size_train']
    globals.batch_size_eval = hyperparameters['batch_size_eval']
    globals.num_workers = hyperparameters['num_workers']
    smooth = hyperparameters['smooth']
    decay = hyperparameters['decay']
    dropout = hyperparameters['dropout']
    lr_decay_factor = hyperparameters['lr_decay_factor']
    patience =   hyperparameters['patience']
    globals.device = hyperparameters['device']
    batch_norm = hyperparameters['batch_norm']
    nheads = hyperparameters['nheads']
    alpha_gat = hyperparameters['alpha_gat']
    best_loss = hyperparameters['best_loss']
    test_mode = hyperparameters['test_mode']
    save_train_graph_path = hyperparameters['save_train_graph_path']
    save_val_graph_path = hyperparameters['save_val_graph_path']
    save_test_graph_path = hyperparameters['save_test_graph_path']
    globals.save_garph_path = hyperparameters['save_garph_path']
    
    utils.set_seed(seed)
    ###################################
    if test_mode:
        test_graph_list = []
        test_node_num = []
        test_edge_num = [] 
        test_toal_pos_to_neg_ratio = []
        test_toal_pos_ratio = []
        test_bin_files = glob.glob(os.path.join(save_test_graph_path, "*.bin"))
        for file_path in test_bin_files:
            graph_list, _ = dgl.load_graphs(file_path)
            test_graph = graph_list[0]
            num_nodes = test_graph.num_nodes()
            num_edges = test_graph.num_edges()
            test_graph.ndata['x'] = torch.ones(num_nodes, dtype=torch.float32)
            test_graph_list.append(test_graph)  # 将所有读取的图添加到列表中
            test_node_num.append(num_nodes)
            test_edge_num.append(num_edges)
            pos_count = (test_graph.edata['y'] == 1).sum().item()
            neg_count = (test_graph.edata['y'] == 0).sum().item()
            pos_to_neg_ratio = pos_count / neg_count if neg_count != 0 else float('inf')
            pos_ratio = pos_count / num_edges
            test_toal_pos_to_neg_ratio.append(pos_to_neg_ratio)
            test_toal_pos_ratio.append(pos_ratio)
        print(f'Number of testing graphs: {len(test_graph_list)}')
        print(f'Number of testing nodes: {np.sum(test_node_num)}')
        print(f'Number of testing edges: {np.sum(test_edge_num)}')
        test_pos_to_neg_ratio = np.mean(test_toal_pos_to_neg_ratio)
        test_pos_ratio = np.mean(test_toal_pos_ratio)
        print(Fore.RED + 'Positive to negative ratio: ' + Style.RESET_ALL, f"{round(test_pos_to_neg_ratio, 4)}")
        print(Fore.RED + 'Positive ratio: ' + Style.RESET_ALL, f'{round(test_pos_ratio, 4)}')
        pos_weight = 1.0 / test_pos_to_neg_ratio
        pos_weight = torch.tensor([pos_weight], dtype=torch.float32).to(globals.device)  # 转换为张量
        globals.test_criterion = models.LayerWiseDiceBalancedCrossEntropyLoss(pos_weight=pos_weight, smooth=smooth)

    else:

        train_graph_list = []
        train_node_num = []
        train_edge_num = []
        train_toal_pos_to_neg_ratio = []
        train_toal_pos_ratio = []
        train_bin_files = glob.glob(os.path.join(save_train_graph_path, "*.bin"))
        # 遍历每个 .bin 文件并加载图
        for file_path in train_bin_files:
            graph_list, _ = dgl.load_graphs(file_path)
            train_graph = graph_list[0]
            num_nodes = train_graph.num_nodes()
            num_edges = train_graph.num_edges()
            train_graph.ndata['x'] = torch.ones(num_nodes, dtype=torch.float32)
            train_graph_list.append(train_graph)  # 将所有读取的图添加到列表中
            train_node_num.append(num_nodes)
            train_edge_num.append(num_edges)
            pos_count = (train_graph.edata['y'] == 1).sum().item()
            neg_count = (train_graph.edata['y'] == 0).sum().item()
            pos_to_neg_ratio = pos_count / neg_count if neg_count != 0 else float('inf')
            pos_ratio = pos_count / num_edges
            train_toal_pos_to_neg_ratio.append(pos_to_neg_ratio)
            train_toal_pos_ratio.append(pos_ratio)
        
        print(f'Number of training graphs: {len(train_graph_list)}')
        print(f'Number of training nodes: {np.sum(train_node_num)}')
        print(f'Number of training edges: {np.sum(train_edge_num)}')
        train_pos_to_neg_ratio = np.mean(train_toal_pos_to_neg_ratio)
        train_pos_ratio = np.mean(train_toal_pos_ratio)
        print(Fore.RED + 'Positive to negative ratio: ' + Style.RESET_ALL, f"{round(train_pos_to_neg_ratio, 4)}")
        print(Fore.RED + 'Positive ratio: ' + Style.RESET_ALL, f'{round(train_pos_ratio, 4)}')
        pos_weight = 1.0 / train_pos_to_neg_ratio
        pos_weight = torch.tensor([pos_weight], dtype=torch.float32).to(globals.device)  # 转换为张量
        globals.train_criterion = models.LayerWiseDiceBalancedCrossEntropyLoss(pos_weight=pos_weight, smooth=smooth)

        # 验证集
        val_graph_list = []
        val_node_num = []
        val_edge_num = [] 
        val_toal_pos_to_neg_ratio = []
        val_toal_pos_ratio = []
        val_bin_files = glob.glob(os.path.join(save_val_graph_path, "*.bin"))
        for file_path in val_bin_files:
            graph_list, _ = dgl.load_graphs(file_path)
            val_graph = graph_list[0]
            num_nodes = val_graph.num_nodes()
            num_edges = val_graph.num_edges()
            val_graph.ndata['x'] = torch.ones(num_nodes, dtype=torch.float32)
            val_graph_list.append(val_graph)  # 将所有读取的图添加到列表中
            val_node_num.append(num_nodes)
            val_edge_num.append(num_edges)
            pos_count = (val_graph.edata['y'] == 1).sum().item()
            neg_count = (val_graph.edata['y'] == 0).sum().item()
            pos_to_neg_ratio = pos_count / neg_count if neg_count != 0 else float('inf')
            pos_ratio = pos_count / num_edges
            val_toal_pos_to_neg_ratio.append(pos_to_neg_ratio)
            val_toal_pos_ratio.append(pos_ratio)
        
        print(f'\nNumber of validating graphs: {len(val_graph_list)}')
        print(f'Number of validating nodes: {np.sum(val_node_num)}')
        print(f'Number of validating edges: {np.sum(val_edge_num)}')
        val_pos_to_neg_ratio = np.mean(val_toal_pos_to_neg_ratio)
        val_pos_ratio = np.mean(val_toal_pos_ratio)
        print(Fore.RED + 'Positive to negative ratio: ' + Style.RESET_ALL, f"{round(val_pos_to_neg_ratio, 4)}")
        print(Fore.RED + 'Positive ratio: ' + Style.RESET_ALL, f'{round(val_pos_ratio, 4)}')
        pos_weight = 1.0 / val_pos_to_neg_ratio
        pos_weight = torch.tensor([pos_weight], dtype=torch.float32).to(globals.device)  # 转换为张量
        globals.val_criterion = models.LayerWiseDiceBalancedCrossEntropyLoss(pos_weight=pos_weight, smooth=smooth)

    ###################################
    print(f'\nDevice: {globals.device}')  
    model_params = {
    'node_features': node_features,
    'edge_features': edge_features,
    'hidden_features': hidden_features,
    'num_layers': num_layers,
    'hidden_edge_scores': hidden_edge_scores,
    'batch_norm': batch_norm,
    'nb_pos_enc': globals.nb_pos_enc,
    'dropout': dropout,
    'nheads': nheads,
    'alpha_gat': alpha_gat
                    }

    model = models.GATmodel(**model_params)
    # model = models.resgated_multidigraph()
    ## model = models.GraphGatedGCNModel() 
    model.to(globals.device)

    print(f'\nNumber of network parameters: {view_model_param(model)}\n')
    print(f'Normalization type : Batch Normalization\n') if batch_norm else print(f'Normalization type : Layer Normalization\n')

    globals.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    scheduler = ReduceLROnPlateau(globals.optimizer, mode='min', factor=lr_decay_factor, patience=patience, min_lr=1e-6, verbose=True)
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(os.path.join(base_path, 'checkpoints')):
        os.makedirs(os.path.join(base_path, 'checkpoints'))

    globals.cluster_cache_path = os.path.join(base_path, f'checkpoints/{out}_cluster_gcn.pkl')
    if os.path.exists(globals.cluster_cache_path):
        os.remove(globals.cluster_cache_path)

    best_model_path = os.path.join(base_path, f'checkpoints/best_model.pth')
    if not test_mode:
        if os.path.exists(best_model_path):
            os.remove(best_model_path)
    
    json_file_path = os.path.join(base_path, f'checkpoints/info.json')
    if not test_mode:
        if os.path.exists(json_file_path):
            os.remove(json_file_path)

    os.environ['WANDB_DIR'] = os.path.dirname(os.path.abspath(__file__))
    try:
        with wandb.init(project='KANGNN_for_Assembly', entity='xielei1203-dalian-university', name=dataset_name, mode=wandb_mode, config=hyperparameters):
            wandb.watch(model, log='all', log_freq=1000)
            
            if test_mode:
                model.load_state_dict(torch.load(best_model_path))
                val_test_data = gen_val_test_batch(test_graph_list)
                model.eval()
                all_test_loss, all_test_fp_rate, all_test_fn_rate = [], [], []
                all_test_acc, all_test_precision, all_test_recall, all_test_f1, all_test_mcc = [], [], [], [], []
                for test_graph_id, test_data_graph in enumerate(val_test_data):
                    print(Fore.RED +'\nTesting on graph: ' + Style.RESET_ALL, f'{test_graph_id}')
                    graph = test_graph_list[test_graph_id]
                    test_results = test_process(model, test_data_graph, graph, test_graph_id)
                    all_test_loss.append(test_results['val_loss'])
                    all_test_fp_rate.append(test_results['val_fp_rate'])
                    all_test_fn_rate.append(test_results['val_fn_rate'])
                    all_test_acc.append(test_results['val_acc'])
                    all_test_precision.append(test_results['val_precision'])
                    all_test_recall.append(test_results['val_recall'])
                    all_test_f1.append(test_results['val_f1'])
                    all_test_mcc.append(test_results['val_mcc'])
                test_loss = np.mean(all_test_loss)
                test_fp_rate = np.mean(all_test_fp_rate)
                test_fn_rate = np.mean(all_test_fn_rate)
                test_acc = np.mean(all_test_acc)
                test_precision = np.mean(all_test_precision)
                test_recall = np.mean(all_test_recall)
                test_f1 = np.mean(all_test_f1)
                test_mcc = np.mean(all_test_mcc)

                current_time = Fore.GREEN + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + Style.RESET_ALL
                print(Fore.RED +'\n===>Testing: ' + Style.RESET_ALL)
                print(f'[{current_time}] Loss: {test_loss:.4f}, fp_rate: {test_fp_rate:.4f}, fn_rate: {test_fn_rate:.4f}')
                print(f'test_acc: {test_acc:.4f}, test_precision: {test_precision:.4f}, test_recall: {test_recall:.4f}, test_f1: {test_f1:.4f}, test_mcc: {test_mcc:.4f}')  
            
            else:

                train_data = gen_train_batch(train_graph_list)
                val_test_data = gen_val_test_batch(val_graph_list)
                
                for epoch in range(0, num_epochs):
                    all_train_loss, all_train_fp_rate, all_train_fn_rate = [], [], []
                    all_train_acc, all_train_precision, all_train_recall, all_train_f1, all_train_mcc = [], [], [], [], []
                    
                    model.train()
                    for train_graph_id, train_data_graph in enumerate(train_data):
                        print(Fore.RED +'\nTraining on graph: ' + Style.RESET_ALL, f'{train_graph_id}')
                        train_results = train_process(model, train_data_graph)
                        
                        all_train_loss.append(train_results['train_loss'])
                        all_train_fp_rate.append(train_results['train_fp_rate'])
                        all_train_fn_rate.append(train_results['train_fn_rate'])
                        all_train_acc.append(train_results['train_acc'])
                        all_train_precision.append(train_results['train_precision'])
                        all_train_recall.append(train_results['train_recall'])
                        all_train_f1.append(train_results['train_f1'])
                        all_train_mcc.append(train_results['train_mcc'])
                    
                    train_loss = np.mean(all_train_loss)
                    train_fp_rate = np.mean(all_train_fp_rate)
                    train_fn_rate = np.mean(all_train_fn_rate)
                    train_acc = np.mean(all_train_acc)
                    train_precision = np.mean(all_train_precision)
                    train_recall = np.mean(all_train_recall)
                    train_f1 = np.mean(all_train_f1)
                    train_mcc = np.mean(all_train_mcc)
                    scheduler.step(train_loss)

                    all_val_loss, all_val_fp_rate, all_val_fn_rate = [], [], []
                    all_val_acc, all_val_precision, all_val_recall, all_val_f1, all_val_mcc = [], [], [], [], []
                    
                    model.eval() 
                    for val_graph_id, val_data_graph in enumerate(val_test_data):
                        print(Fore.RED +'\nValidating on graph: ' + Style.RESET_ALL, f'{val_graph_id}')
                        val_results = val_process(model, val_data_graph)
                        
                        all_val_loss.append(val_results['val_loss'])
                        all_val_fp_rate.append(val_results['val_fp_rate'])
                        all_val_fn_rate.append(val_results['val_fn_rate'])
                        all_val_acc.append(val_results['val_acc'])
                        all_val_precision.append(val_results['val_precision'])
                        all_val_recall.append(val_results['val_recall'])
                        all_val_f1.append(val_results['val_f1'])
                        all_val_mcc.append(val_results['val_mcc'])
                    
                    val_loss = np.mean(all_val_loss)
                    val_fp_rate = np.mean(all_val_fp_rate)
                    val_fn_rate = np.mean(all_val_fn_rate)
                    val_acc = np.mean(all_val_acc)
                    val_precision = np.mean(all_val_precision)
                    val_recall = np.mean(all_val_recall)
                    val_f1 = np.mean(all_val_f1)
                    val_mcc = np.mean(all_val_mcc)

                    if epoch % 1 == 0:
                        current_time = Fore.GREEN + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + Style.RESET_ALL
                        print(Fore.RED + '\n===>TRAINING: '+ Style.RESET_ALL, f'Epoch = {epoch}')
                        print(f'[{current_time}] Loss: {train_loss:.4f}, fp_rate: {train_fp_rate:.4f}, fn_rate: {train_fn_rate:.4f}')
                        print(f'train_acc: {train_acc:.4f}, train_precision: {train_precision:.4f}, train_recall: {train_recall:.4f}, train_f1: {train_f1:.4f}, train_mcc: {train_mcc:.4f}')
                    
                        print(Fore.RED +'\n===>Validating: ' + Style.RESET_ALL, f'Epoch = {epoch}')
                        print(f'[{current_time}] Loss: {val_loss:.4f}, fp_rate: {val_fp_rate:.4f}, fn_rate: {val_fn_rate:.4f}')
                        print(f'val_acc: {val_acc:.4f}, val_precision: {val_precision:.4f}, val_recall: {val_recall:.4f}, val_f1: {val_f1:.4f}, val_mcc: {val_mcc:.4f}')
                    
                    info = {
                            'dataset_name': dataset_name,
                            'epoch': epoch,
                            'val_loss': val_loss,
                            'val_fp_rate': val_fp_rate,
                            'val_fn_rate': val_fn_rate,
                            'val_acc': val_acc,
                            'val_precision': val_precision,
                            'val_recall': val_recall,
                            'val_f1': val_f1,
                            'val_mcc': val_mcc,
                            }
                    if val_loss < best_loss:
                        best_loss = val_loss
                        torch.save(model.state_dict(), best_model_path)
                        with open(json_file_path, 'w') as json_file:
                                json.dump(info, json_file, indent=4)

                    try:
                        wandb.log({'train_loss': train_loss, \
                                    'train_fp_rate': train_fp_rate, 'train_fn_rate': train_fn_rate, \
                                    'train_accuracy': train_acc, 'train_precision': train_precision, \
                                    'train_recall': train_recall, 'train_f1': train_f1, 'train_mcc': train_mcc, \
                                    'val_loss': val_loss, \
                                    'val_fp_rate': val_fp_rate, 'val_fn_rate': val_fn_rate, \
                                    'val_accuracy': val_acc, 'val_precision': val_precision, \
                                    'val_recall': val_recall, 'val_f1': val_f1, 'val_mcc': val_mcc, \
                                    })
                        
                    except Exception:
                        print(f'WandB exception occured!')


    except KeyboardInterrupt:
        print("Keyboard Interrupt...")
        print("Exiting...")
    wandb.finish()

def train_process(model, train_data_graph):
    if globals.batch_size_train <= 1: # train with full graph 
        g = add_positional_encoding(train_data_graph, globals.nb_pos_enc) # add in_degree and out_degree positional encoding
        g = g.to(globals.device)
        x = g.ndata['x'].to(globals.device)
        e = g.edata['e'].to(globals.device)
        pe = g.ndata['pe'].to(globals.device) # 位置编码
        pe_in = g.ndata['in_deg'].unsqueeze(1).to(globals.device)
        pe_out = g.ndata['out_deg'].unsqueeze(1).to(globals.device)
        pe = torch.cat((pe_in, pe_out, pe), dim=1) 
        adj_torch = generate_adj_matrix(g, globals.device)
        edge_predictions = model(g, x, e, pe, adj_torch) 
        edge_predictions = edge_predictions.squeeze(-1)
        edge_labels = g.edata['y'].to(globals.device)
        loss = globals.criterion(edge_predictions, edge_labels)
        globals.optimizer.zero_grad()
        loss.backward()
        globals.optimizer.step()
        train_loss = loss.item()
        TP, TN, FP, FN = utils.calculate_tfpn(edge_predictions, edge_labels)
        acc, precision, recall, f1, mcc =  utils.calculate_metrics(TP, TN, FP, FN)
        try:
            fp_rate = FP / (FP + TN)
        except ZeroDivisionError:
            fp_rate = 0.0
        try:
            fn_rate = FN / (FN + TP)
        except ZeroDivisionError:
            fn_rate = 0.0
        train_fp_rate = fp_rate
        train_fn_rate = fn_rate
        train_acc = acc
        train_precision = precision
        train_recall = recall
        train_f1 = f1
        train_mcc = mcc

    else: # train with mini-batch
        # remove Metis clusters to force new clusters
        # try:
        #     os.remove(globals.cluster_cache_path)
        # except:
        #     pass 

        # # Run Metis
        # g = train_data_graph.long()
        # num_clusters = torch.LongTensor(1).random_(globals.num_parts_metis_train-100,globals.num_parts_metis_train+100).item() # DEBUG!!!
        # sampler = dgl.dataloading.ClusterGCNSampler(g, num_clusters, cache_path=globals.cluster_cache_path) 
        # dataloader = dgl.dataloading.DataLoader(g, torch.arange(num_clusters), sampler, batch_size=globals.batch_size_train, shuffle=True, drop_last=False, num_workers=globals.num_workers)
        
        # For loop over all mini-batch in the graph
        running_loss, running_fp_rate, running_fn_rate = [], [], []
        running_acc, running_precision, running_recall, running_f1, running_mcc = [], [], [], [], []
        with train_data_graph.enable_cpu_affinity():
            for sub_g in train_data_graph:
                sub_g = add_positional_encoding(sub_g, globals.nb_pos_enc)
                sub_g = sub_g.to(globals.device)
                # x = sub_g.ndata['x'].to(globals.device)
                x = sub_g.ndata['x'].unsqueeze(1).to(globals.device)
                e = sub_g.edata['e'].to(globals.device)
                # e = sub_g.edata['e'][:, 0].unsqueeze(1).to(globals.device)
                pe = sub_g.ndata['pe'].to(globals.device)
                degree_in = sub_g.ndata['in_deg'].unsqueeze(1).to(globals.device)
                degree_out = sub_g.ndata['out_deg'].unsqueeze(1).to(globals.device)
                # degree = torch.cat((degree_in, degree_out), dim=1)
                pe = torch.cat((degree_in, degree_out, pe), dim=1)
                adj_torch = generate_adj_matrix(sub_g, globals.device)
                edge_predictions = model(sub_g, x, e, pe, adj_torch) 
                edge_predictions = edge_predictions.squeeze(-1)
                edge_labels = sub_g.edata['y'].to(globals.device)
                loss = globals.train_criterion(edge_predictions, edge_labels)
                globals.optimizer.zero_grad()
                loss.backward()
                globals.optimizer.step()
                running_loss.append(loss.item())
                TP, TN, FP, FN = utils.calculate_tfpn(edge_predictions, edge_labels)
                acc, precision, recall, f1, mcc =  utils.calculate_metrics(TP, TN, FP, FN)
                try:
                    fp_rate = FP / (FP + TN)
                except ZeroDivisionError:
                    fp_rate = 0.0
                try:
                    fn_rate = FN / (FN + TP)
                except ZeroDivisionError:
                    fn_rate = 0.0
                running_fp_rate.append(fp_rate)
                running_fn_rate.append(fn_rate)
                running_acc.append(acc)
                running_precision.append(precision)
                running_recall.append(recall)
                running_f1.append(f1)
                running_mcc.append(mcc)

        # Average over all mini-batch in the graph
        train_loss = np.mean(running_loss)
        train_fp_rate = np.mean(running_fp_rate)
        train_fn_rate = np.mean(running_fn_rate)
        train_acc = np.mean(running_acc)
        train_precision = np.mean(running_precision)
        train_recall = np.mean(running_recall)
        train_f1 = np.mean(running_f1)
        train_mcc = np.mean(running_mcc)

    return {'train_loss': train_loss, 'train_fp_rate': train_fp_rate, 
            'train_fn_rate': train_fn_rate, 'train_acc': train_acc, 
            'train_precision': train_precision, 'train_recall': train_recall,
            'train_f1': train_f1, 'train_mcc': train_mcc}

def val_process(model, val_data_graph):
    with torch.no_grad():
        if globals.batch_size_eval <= 1: # full graph 
            g = add_positional_encoding(val_data_graph, globals.nb_pos_enc)
            g = g.to(globals.device)
            x = g.ndata['x'].to(globals.device)
            e = g.edata['e'].to(globals.device)
            pe = g.ndata['pe'].to(globals.device)
            pe_in = g.ndata['in_deg'].unsqueeze(1).to(globals.device)
            pe_out = g.ndata['out_deg'].unsqueeze(1).to(globals.device)
            pe = torch.cat((pe_in, pe_out, pe), dim=1)
            adj_torch = generate_adj_matrix(g, globals.device)
            edge_predictions = model(g, x, e, pe, adj_torch)
            edge_predictions = edge_predictions.squeeze(-1)
            edge_labels = g.edata['y'].to(globals.device)
            loss = globals.criterion(edge_predictions, edge_labels)
            val_loss = loss.item()
            TP, TN, FP, FN = utils.calculate_tfpn(edge_predictions, edge_labels)
            acc, precision, recall, f1, mcc =  utils.calculate_metrics(TP, TN, FP, FN)
            try:
                fp_rate = FP / (FP + TN)
            except ZeroDivisionError:
                fp_rate = 0.0
            try:
                fn_rate = FN / (FN + TP)
            except ZeroDivisionError:
                fn_rate = 0.0
            val_fp_rate = fp_rate
            val_fn_rate = fn_rate
            val_acc = acc
            val_precision = precision
            val_recall = recall
            val_f1 = f1
            val_mcc = mcc

        else: # mini-batch

            # remove Metis clusters to force new clusters
            # try:
            #     os.remove(globals.cluster_cache_path)
            # except:
            #     pass 

            # # Run Metis
            # g = val_data_graph.long()
            # sampler = dgl.dataloading.ClusterGCNSampler(g, globals.num_parts_metis_eval, cache_path=globals.cluster_cache_path) 
            # dataloader = dgl.dataloading.DataLoader(g, torch.arange(globals.num_parts_metis_eval), sampler, batch_size=globals.batch_size_eval, shuffle=True, drop_last=False, num_workers=globals.num_workers)
            
            # For loop over all mini-batch in the graph
            running_loss, running_fp_rate, running_fn_rate = [], [], []
            running_acc, running_precision, running_recall, running_f1, running_mcc = [], [], [], [], []
            with val_data_graph.enable_cpu_affinity():
                for sub_g in val_data_graph:
                    sub_g = add_positional_encoding(sub_g, globals.nb_pos_enc)
                    sub_g = sub_g.to(globals.device)
                    # x = sub_g.ndata['x'].to(globals.device)
                    x = sub_g.ndata['x'].unsqueeze(1).to(globals.device)
                    e = sub_g.edata['e'].to(globals.device)
                    # e = sub_g.edata['e'][:, 0].unsqueeze(1).to(globals.device)
                    pe = sub_g.ndata['pe'].to(globals.device)
                    degree_in = sub_g.ndata['in_deg'].unsqueeze(1).to(globals.device)
                    degree_out = sub_g.ndata['out_deg'].unsqueeze(1).to(globals.device)
                    # degree = torch.cat((degree_in, degree_out), dim=1)
                    pe = torch.cat((degree_in, degree_out, pe), dim=1)
                    adj_torch = generate_adj_matrix(sub_g, globals.device)
                    edge_predictions = model(sub_g, x, e, pe, adj_torch) 
                    edge_predictions = edge_predictions.squeeze(-1)
                    edge_labels = sub_g.edata['y'].to(globals.device)
                    loss = globals.val_criterion(edge_predictions, edge_labels)
                    running_loss.append(loss.item())
                    TP, TN, FP, FN = utils.calculate_tfpn(edge_predictions, edge_labels)
                    acc, precision, recall, f1, mcc =  utils.calculate_metrics(TP, TN, FP, FN)
                    try:
                        fp_rate = FP / (FP + TN)
                    except ZeroDivisionError:
                        fp_rate = 0.0
                    try:
                        fn_rate = FN / (FN + TP)
                    except ZeroDivisionError:
                        fn_rate = 0.0
                    running_fp_rate.append(fp_rate)
                    running_fn_rate.append(fn_rate)
                    running_acc.append(acc)
                    running_precision.append(precision)
                    running_recall.append(recall)
                    running_f1.append(f1)
                    running_mcc.append(mcc)

            # Average over all mini-batch in the graph
            val_loss = np.mean(running_loss)
            val_fp_rate = np.mean(running_fp_rate)
            val_fn_rate = np.mean(running_fn_rate)
            val_acc = np.mean(running_acc)
            val_precision = np.mean(running_precision)
            val_recall = np.mean(running_recall)
            val_f1 = np.mean(running_f1)
            val_mcc = np.mean(running_mcc)

        return {'val_loss': val_loss, 'val_fp_rate': val_fp_rate, 
                'val_fn_rate': val_fn_rate, 'val_acc': val_acc, 
                'val_precision': val_precision, 'val_recall': val_recall,
                'val_f1': val_f1, 'val_mcc': val_mcc}
    
def test_process(model, test_data_graph, graph, test_graph_id):
    with torch.no_grad():
        if globals.batch_size_eval <= 1: # full graph 
            g = add_positional_encoding(test_data_graph, globals.nb_pos_enc)
            g = g.to(globals.device)
            x = g.ndata['x'].to(globals.device)
            e = g.edata['e'].to(globals.device)
            pe = g.ndata['pe'].to(globals.device)
            pe_in = g.ndata['in_deg'].unsqueeze(1).to(globals.device)
            pe_out = g.ndata['out_deg'].unsqueeze(1).to(globals.device)
            pe = torch.cat((pe_in, pe_out, pe), dim=1)
            adj_torch = generate_adj_matrix(g, globals.device)
            edge_predictions = model(g, x, e, pe, adj_torch)
            edge_predictions = edge_predictions.squeeze(-1)
            edge_labels = g.edata['y'].to(globals.device)
            loss = globals.criterion(edge_predictions, edge_labels)
            test_loss = loss.item()
            TP, TN, FP, FN = utils.calculate_tfpn(edge_predictions, edge_labels)
            acc, precision, recall, f1, mcc =  utils.calculate_metrics(TP, TN, FP, FN)
            try:
                fp_rate = FP / (FP + TN)
            except ZeroDivisionError:
                fp_rate = 0.0
            try:
                fn_rate = FN / (FN + TP)
            except ZeroDivisionError:
                fn_rate = 0.0
            test_fp_rate = fp_rate
            test_fn_rate = fn_rate
            test_acc = acc
            test_precision = precision
            test_recall = recall
            test_f1 = f1
            test_mcc = mcc

        else: # mini-batch

            # remove Metis clusters to force new clusters
            # try:
            #     os.remove(globals.cluster_cache_path)
            # except:
            #     pass 

            # # Run Metis
            # g = val_data_graph.long()
            # sampler = dgl.dataloading.ClusterGCNSampler(g, globals.num_parts_metis_eval, cache_path=globals.cluster_cache_path) 
            # dataloader = dgl.dataloading.DataLoader(g, torch.arange(globals.num_parts_metis_eval), sampler, batch_size=globals.batch_size_eval, shuffle=True, drop_last=False, num_workers=globals.num_workers)
            
            # For loop over all mini-batch in the graph
            running_loss, running_fp_rate, running_fn_rate = [], [], []
            running_acc, running_precision, running_recall, running_f1, running_mcc = [], [], [], [], []
            
            # 初始化一个全图大小的score张量
            graph = graph.to(globals.device)  # 将图移动到目标设备
            full_graph_score = torch.zeros(graph.num_edges(), dtype=torch.float32, device=globals.device)
            
            with test_data_graph.enable_cpu_affinity():
                for sub_g in test_data_graph:
                    sub_g = add_positional_encoding(sub_g, globals.nb_pos_enc)
                    sub_g = sub_g.to(globals.device)
                    # x = sub_g.ndata['x'].to(globals.device)
                    x = sub_g.ndata['x'].unsqueeze(1).to(globals.device)
                    e = sub_g.edata['e'].to(globals.device)
                    # e = sub_g.edata['e'][:, 0].unsqueeze(1).to(globals.device)
                    pe = sub_g.ndata['pe'].to(globals.device)
                    degree_in = sub_g.ndata['in_deg'].unsqueeze(1).to(globals.device)
                    degree_out = sub_g.ndata['out_deg'].unsqueeze(1).to(globals.device)
                    # degree = torch.cat((degree_in, degree_out), dim=1)
                    pe = torch.cat((degree_in, degree_out, pe), dim=1)
                    adj_torch = generate_adj_matrix(sub_g, globals.device)
                    edge_predictions = model(sub_g, x, e, pe, adj_torch) 
                    edge_predictions = edge_predictions.squeeze(-1)

                    # 将子图的边索引和score结果写回到全图的相应位置
                    full_graph_score[sub_g.edata[dgl.EID]] = edge_predictions

                    edge_labels = sub_g.edata['y'].to(globals.device)
                    loss = globals.test_criterion(edge_predictions, edge_labels)
                    running_loss.append(loss.item())
                    TP, TN, FP, FN = utils.calculate_tfpn(edge_predictions, edge_labels)
                    acc, precision, recall, f1, mcc =  utils.calculate_metrics(TP, TN, FP, FN)
                    try:
                        fp_rate = FP / (FP + TN)
                    except ZeroDivisionError:
                        fp_rate = 0.0
                    try:
                        fn_rate = FN / (FN + TP)
                    except ZeroDivisionError:
                        fn_rate = 0.0
                    running_fp_rate.append(fp_rate)
                    running_fn_rate.append(fn_rate)
                    running_acc.append(acc)
                    running_precision.append(precision)
                    running_recall.append(recall)
                    running_f1.append(f1)
                    running_mcc.append(mcc)
                # 在循环结束后，full_graph_score 就包含了全图的边得分
                nor_score = (torch.sigmoid(full_graph_score) >= 0.4).float()
                # nor_score = torch.round(torch.sigmoid(full_graph_score))
                num_ones = (nor_score == 1).sum().item()
                print(f"Number of edges predicted as 1: {num_ones}")
                graph.edata['score'] = nor_score
                # # 如果是测试集且提供了保存路径，保存图
                save_path = f"{globals.save_garph_path}/test_{test_graph_id}_graph.bin"
                dgl.save_graphs(save_path, graph)
                print(f"Graph saved to {save_path}")
            # Average over all mini-batch in the graph
            test_loss = np.mean(running_loss)
            test_fp_rate = np.mean(running_fp_rate)
            test_fn_rate = np.mean(running_fn_rate)
            test_acc = np.mean(running_acc)
            test_precision = np.mean(running_precision)
            test_recall = np.mean(running_recall)
            test_f1 = np.mean(running_f1)
            test_mcc = np.mean(running_mcc)

        return {'val_loss': test_loss, 'val_fp_rate': test_fp_rate, 
                'val_fn_rate': test_fn_rate, 'val_acc': test_acc, 
                'val_precision': test_precision, 'val_recall': test_recall,
                'val_f1': test_f1, 'val_mcc': test_mcc}

if __name__ == '__main__':
    train()
    

