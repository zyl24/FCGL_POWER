import os
import pickle
import random
import numpy as np
import torch
from openfgl.config import args as openfgl_args
from openfgl.data.distributed_dataset_loader import FGLDataset


def load_fcgl_dataset(root, dataset="Cora", num_clients=3, classes_per_task=2, shuffle_task=False):
    print(f"loading {dataset} dataset...")
    openfgl_args.root = root
    openfgl_args.scenario = "subgraph_fl"
    openfgl_args.simulation_mode = "subgraph_fl_louvain"
    openfgl_args.dataset = [dataset]
    openfgl_args.num_clients = num_clients
    data = FGLDataset(openfgl_args)
    
    
    
    if shuffle_task:
        task_dir = os.path.join(data.processed_dir, f"task_{classes_per_task}_shuffle")
    else:
        task_dir = os.path.join(data.processed_dir, f"task_{classes_per_task}_no_shuffle")
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
            
    if dataset in ["Cora", "CiteSeer", "Computers", "Physics", "Actor"]:
        train_, val_, test_ = 0.2, 0.4, 0.4
    elif dataset in ["ogbn-arxiv", "Flickr"]:
        train_, val_, test_ = 0.6, 0.2, 0.2
    elif dataset == "Squirrel":
        train_, val_, test_ = 0.48, 0.32, 0.2
    elif dataset == "Roman-empire":
        train_, val_, test_ = 0.5, 0.25, 0.25
        
        
    processed_data = {client_id: {"data": None,
                                "task": None} for client_id in range(num_clients)}
    
    known_class_list = []
    
    for client_id in range(num_clients):
        local_data = data.local_data[client_id]
        task_file_path = os.path.join(task_dir, f"client_{client_id}.pkl")
        processed_data[client_id]["data"] = local_data
        
        if not os.path.exists(task_file_path):
            task_file = class_incremental(local_data, classes_per_task, train_, val_, test_, shuffle_task=shuffle_task)
            with open(task_file_path, 'wb') as file:
                pickle.dump(task_file, file)
                
        with open(task_file_path, 'rb') as file:
            task_file = pickle.load(file)

        processed_data[client_id]["task"] = task_file
        
        for task_i in task_file:
            client_i_task_i_mask = task_i["train_mask"] | task_i["val_mask"] | task_i["test_mask"]
            client_i_task_i_known_classes = torch.unique(local_data.y[client_i_task_i_mask])
            known_class_list.append(client_i_task_i_known_classes)

        
        print(f"client {client_id} has {len(processed_data[client_id]['task'])} tasks.")

    known_class = torch.unique(torch.hstack(known_class_list))
    num_remained_classes = known_class.shape[0]

    in_dim = data.global_data.x.shape[1]
    out_dim = num_remained_classes
    
    if num_remained_classes != data.global_data.num_global_classes:
        print(f"ATTENTION!!! FCGL DROPS {data.global_data.num_global_classes - num_remained_classes} CLASS(ES).")
        
    return processed_data, in_dim, out_dim, task_dir
        
        



def class_incremental(data, classes_per_task, train_split, val_split, test_split, shuffle_task=False):
    num_nodes = data.x.shape[0]
    num_classes = data.y.max().item() + 1
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    for class_i in range(num_classes):
        class_i_node_mask = data.y == class_i
        num_class_i_nodes = class_i_node_mask.sum().item()
        
        class_i_node_list = torch.where(class_i_node_mask)[0].numpy()
        np.random.shuffle(class_i_node_list)

        train_end = int(train_split * num_class_i_nodes)
        val_end = int((train_split + val_split) * num_class_i_nodes)
        
        train_indices = class_i_node_list[:train_end]
        val_indices = class_i_node_list[train_end:val_end]
        test_indices = class_i_node_list[val_end:num_class_i_nodes]
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
    
    num_tasks = (num_classes + classes_per_task - 1) // classes_per_task
    
    label_to_task = {}
    
    drop_signal = False
    
    shuffled_classes = list(range(num_classes))
    if shuffle_task:
        random.shuffle(shuffled_classes)
    
    for task_id in range(num_tasks):
        left = task_id * classes_per_task
        right = min((task_id + 1) * classes_per_task, num_classes)
        if right < (task_id + 1) * classes_per_task:
            drop_signal = True        
    
        for ptr in range(left, right):
            label_to_task[shuffled_classes[ptr]] = task_id
    
    if drop_signal:
        num_tasks -= 1

    tasks = [{"train_mask": torch.zeros_like(train_mask).bool(), 
              "val_mask": torch.zeros_like(val_mask).bool(), 
              "test_mask": torch.zeros_like(test_mask).bool()} for _ in range(num_tasks)] 

    for class_i in range(num_classes):
        class_i_train = train_mask & (data.y == class_i)
        class_i_val = val_mask & (data.y == class_i)
        class_i_test = test_mask & (data.y == class_i)
        task_i = label_to_task[class_i]
        if task_i == num_tasks:
            continue
        tasks[task_i]["train_mask"] = tasks[task_i]["train_mask"] | class_i_train
        tasks[task_i]["val_mask"] = tasks[task_i]["val_mask"] | class_i_val
        tasks[task_i]["test_mask"] = tasks[task_i]["test_mask"] | class_i_test

    
    np.random.shuffle(tasks)
    
    return tasks
