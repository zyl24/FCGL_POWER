import copy
import random
import subprocess
import sys

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, dense_to_sparse, add_self_loops, remove_self_loops
import os
import torch.nn.functional as F


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.set_num_threads(1)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    

def start(args, fcgl_dataset, clients, server, message_pool, device):
    if args.global_eval:
        assert not args.isolate_mode
    
    # data analysis
    print("-"*50)
    for client_id in range(args.num_clients):
        result_list = []
        for task_id in range(len(clients[client_id].data["task"])):
            label = clients[client_id].data["data"].y
            task = clients[client_id].data["task"][task_id]
            node_mask = task["train_mask"] | task["val_mask"] | task["test_mask"]
            result = torch.unique(label[node_mask]).cpu().tolist()
            result_list.append(str(result))
        print(f"client {client_id} tasks: {'=>'.join(result_list)}")
        

    # start train
    print("-"*50)
    num_tasks = len(fcgl_dataset[0]["task"])
    args.num_tasks = num_tasks
        
    FL_acc_matrix = torch.zeros(size=(num_tasks, num_tasks)).to(device)

    for task_id in range(num_tasks): 
        FL_acc_matrix[task_id, :] = 0
        message_pool["task_id"] = task_id
        for client_id in range(args.num_clients):
            clients[client_id].task_start(task_id)
        server.task_start(task_id)
        for round_id in range(args.num_rounds_per_task):
            message_pool["round_id"] = round_id
            server.send_message()
            for client_id in range(args.num_clients):
                clients[client_id].execute(task_id)
                clients[client_id].send_message(task_id)        
            server.execute()
        
        for eval_task_id in range(0, task_id+1):
            total_nodes = 0
            for client_id in range(args.num_clients):
                client_acc = clients[client_id].evaluate(task_id=eval_task_id, use_global=args.global_eval)
                num_nodes = clients[client_id].data["task"][eval_task_id]["test_mask"].sum()
                FL_acc_matrix[task_id, eval_task_id] += client_acc * num_nodes
                total_nodes += num_nodes
            FL_acc_matrix[task_id, eval_task_id] /= total_nodes
            print(f"[Task {task_id} Finish] Global Accuracy on Task {eval_task_id}: {FL_acc_matrix[task_id, eval_task_id]:.2f}")
        print(FL_acc_matrix)
        aa = AA(FL_acc_matrix, T=task_id+1)
        af = AF(FL_acc_matrix, T=task_id+1)
        print(f"[Task {task_id} Finish] Global AA: {aa:.2f}\tGlobal AF: {af:.2f}")
        for client_id in range(args.num_clients):
            clients[client_id].task_done(task_id)
        server.task_done(task_id)
        print("-"*50)
    
    return aa, af
        
        
        
        
        

def load_clients_server(args, fcgl_dataset, device):
    message_pool = {}
    
    
    if args.method == "ours":
        from flcore.Ours import OursClient, OursServer
        clients = [OursClient(args, client_id, fcgl_dataset[client_id], message_pool, device) for client_id in range(args.num_clients)]
        server = OursServer(args, message_pool, device)
    
    return clients, server, message_pool
    
    
def edge_masking(edge_index, handled, device):
    num_nodes = edge_index.max().item()+1
    node_mask = torch.zeros(num_nodes, dtype=torch.bool).to(device)
    for node in handled:
        node_mask[node] = True
    mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    edge_index = edge_index[:, mask].to(device)
    self_loop_indices = torch.tensor([[node, node] for node in handled], dtype=torch.long).t().to(device)
    edge_index = torch.cat([edge_index, self_loop_indices], dim=1).to(device)
    edge_index = coalesce(edge_index)
    return edge_index



def accuracy(logits, label):
    pred = logits.max(1)[1]
    correct = (pred == label).sum()
    total = label.shape[0]
    return (correct / total)*100
    

def AA(M_acc, T=None):
    if T is None:
        T = M_acc.size(0)
    result = 0
    for i in range(0, T):
        result += M_acc[T-1, i]
    result /= T
    return result

def AF(M_acc, T=None):
    if T is None:
        T = M_acc.size(0)
    if T == 1: # single task
        return -1 # error
    result = 0
    for i in range(0, T-1):
        forgetting = M_acc[i,i] - M_acc[T-1,i]
        result += forgetting
    result /= T-1   
    return result



def update_buffer(buffer, replay, task, task_data, embedding, num_samples_per_class):
    if num_samples_per_class == 0:
        return buffer
    
    
    if replay == "CM":
        new_buffer = update_converage_max(buffer, task, task_data, embedding, num_samples_per_class)
    
    return new_buffer



def isolate_graph(x, y):
    edge_index = torch.empty((2, 0), dtype=torch.long)
    return Data(x=x, edge_index=edge_index.to(x.device), y=y)



def update_random(buffer, task, task_data, num_samples_per_class):
    train_mask = task["train_mask"]
    train_mask = train_mask.to(task_data.y.device)
    all_classes = torch.unique(task_data.y[train_mask])
    x = []
    y = []
    for class_i in all_classes:
        class_i_train_mask = train_mask & (task_data.y == class_i)
        class_i_train_list = class_i_train_mask.nonzero().squeeze().tolist()
        if type(class_i_train_list) is not list:
            class_i_train_list = [class_i_train_list]
        selected_nodes = random.sample(class_i_train_list, num_samples_per_class)
        x.append(task_data.x[selected_nodes])
        y.append(task_data.y[selected_nodes])
    
    added_x = torch.cat(x)
    added_y = torch.cat(y)
    
    if buffer["x"] is None:
        new_buffer = {"x": added_x, "y": added_y}
    else:
        new_buffer = {"x": torch.cat((buffer["x"], added_x)), "y": torch.cat((buffer["y"], added_y))}
    return new_buffer




def update_converage_max(buffer, task, task_data, embedding, num_samples_per_class, distance=0.1):
    replay = {}
    train_mask = task["train_mask"]
    train_mask = train_mask.to(task_data.y.device)
    all_classes = torch.unique(task_data.y[train_mask])
    x = []
    y = []
    
    replay_buffer = None

    for class_i in all_classes:
        class_i_train_mask = train_mask & (task_data.y == class_i)
        class_i_train_list = class_i_train_mask.nonzero().squeeze().tolist()
        if type(class_i_train_list) is not list:
            class_i_train_list = [class_i_train_list]
            

        count = 0
        memory = []
        temp_memory = []
        cover = []
        
        dist_matrix = torch.cdist(embedding[class_i_train_mask], embedding[class_i_train_mask], p=2)
        distances_mean = torch.mean(dist_matrix)
        dist_bin_matrix = torch.where(dist_matrix<distances_mean.item()* distance,1,0)

        temp_dist_bin_matrix = copy.deepcopy(dist_bin_matrix)
        while count < num_samples_per_class:
            ind = (torch.sum(temp_dist_bin_matrix,0)).argmax()
            memory.append(class_i_train_list[ind])
            temp_memory.append(ind)
            target_dist_matrix = temp_dist_bin_matrix[ind,:]
            new_cover = torch.where(target_dist_matrix==1)[0]
            cover = list(set(cover) | set(new_cover))
            temp_dist_bin_matrix[new_cover,:] = 0
            temp_dist_bin_matrix[:,new_cover] = 0

            #reset
            if len(cover) >= len(class_i_train_list) * 0.9:
                cover = temp_memory
                temp_dist_bin_matrix = copy.deepcopy(dist_bin_matrix)
                temp_dist_bin_matrix[cover,:] = 0
                temp_dist_bin_matrix[:,cover] = 0

            count += 1
        
        memory = torch.from_numpy(np.array(memory))

        replay[class_i] = memory
        
        
        replay_buffer = replay[class_i][:num_samples_per_class]
        x.append(task_data.x[replay_buffer])
        y.append(task_data.y[replay_buffer])
        
        
    added_x = torch.cat(x)
    added_y = torch.cat(y)
    
    if buffer["x"] is None:
        new_buffer = {"x": added_x, "y": added_y}
    else:
        new_buffer = {"x": torch.cat((buffer["x"], added_x)), "y": torch.cat((buffer["y"], added_y))}
    return new_buffer




def run_subprocess(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output, end='')
            sys.stdout.flush()

    sys.exit(0)
    
    
def construct_self_loop_graph(x,y=None):
    edge_index = torch.empty((2, 0), dtype=torch.long).to(x.device)
    if y is not None:
        return Data(x=x, edge_index=edge_index, y=y.to(x.device))
    else:
        return Data(x=x, edge_index=edge_index)


def construct_knn_graph(x, k=1):
    adj_logits = F.sigmoid(torch.mm(x, x.T))
    adjacency_matrix = torch.zeros_like(adj_logits)
    topk_values, topk_indices = torch.topk(adj_logits, k=k, dim=1)
    for i in range(x.shape[0]):
        adjacency_matrix[i, topk_indices[i]] = 1
    adjacency_matrix = adjacency_matrix + adjacency_matrix.t()
    adjacency_matrix[adjacency_matrix > 1] = 1
    adjacency_matrix.fill_diagonal_(1)
    edge = adjacency_matrix.long()
    edge_index, _ = dense_to_sparse(edge)
    edge_index = remove_self_loops(edge_index)[0]
    data = Data(x=x, edge_index=edge_index)
    
    
    return data
    

def check_client_data(args, clients):
    label = []
    counter = []
    for client_id in range(args.num_clients):
        client = clients[client_id]
        for task_id in range(len(client.data['task'])):
            task = client.data["task"][task_id]
            whole_data = client.data["data"]
            selected_index = task["train_mask"] | task["val_mask"] | task["test_mask"]
            task_y = whole_data.y[selected_index]
            for i in torch.unique(task_y):
                label.append(i.item())
                counter.append((task_y == i).sum().item())
                print(f"client {client_id} task {task_id}: class {i.item()} = {(task_y == i).sum().item()}")
    
    print(label)
    print(counter)
    
    