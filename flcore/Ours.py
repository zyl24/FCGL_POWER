import os
import copy
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from flcore.base import BaseClient, BaseServer
from model import load_model
from utils import edge_masking
from torch_geometric.data import Data
from openfgl.utils.basic_utils import idx_to_mask_tensor
from utils import accuracy, update_buffer, isolate_graph,  construct_knn_graph, construct_self_loop_graph





def weights_init(m):
    if hasattr(m, "weight"):
        m.weight.data.uniform_(-0.5, 0.5)
    if hasattr(m, "bias"):
        m.bias.data.uniform_(-0.5, 0.5)



class GEModel(nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super(GEModel, self).__init__()
        self.lin1 = nn.Linear(input_dim, 128)
        self.lin2 = nn.Linear(128, 128)
        self.lin3 = nn.Linear(128, 64)
        self.lin4 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        z1 = self.lin1(x)
        z1 = F.relu(z1)
        z2 = self.lin2(z1)
        z2 = F.relu(z2)
        z3 = self.lin3(z2)
        z3 = F.relu(z3)
        logits = self.lin4(z3)
        return logits
        
    

class OursClient(BaseClient):
    
    def __init__(self, args, client_id, data, message_pool, device):
        super(OursClient, self).__init__(args, client_id, data, message_pool, device)
        self.global_model = load_model(name=args.model, input_dim=args.input_dim, hid_dim=args.hid_dim, output_dim=args.output_dim, dropout=args.dropout).to(self.device)
        self.local_model = load_model(name=args.model, input_dim=args.input_dim, hid_dim=args.hid_dim, output_dim=args.output_dim, dropout=args.dropout).to(self.device)
        self.ge_model = GEModel(input_dim=args.input_dim, output_dim=args.output_dim).to(device)
        self.optim = Adam(self.local_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()
        self.cache = {}
        self.local_buffer = {"x": None, "y":None}
        self.proto_grad = None
        
        
     
    def get_task_num_samples(self, task_id):
        task = self.data["task"][task_id]
        task_mask = task["train_mask"] | task["val_mask"] | task["test_mask"]
        return task_mask.sum()   
    
        
    def execute(self, task_id):
        if "ge" in self.message_pool["server"]:
            with torch.no_grad():
                for (local_ge_param, global_ge_param) in zip(self.ge_model.parameters(), self.message_pool["server"]["ge"]):
                    local_ge_param.data.copy_(global_ge_param)
            print(f"[client {self.client_id}] local gradient encode model is initialized.")
                    
        whole_data = self.data["data"].to(self.device)
        task = self.data["task"][task_id]
        
        if task_id not in self.cache:
            task_data = self.task_data(task_id, whole_data, task)
            self.cache[task_id] = task_data
            
        task_data = self.cache[task_id]
        
        
        if self.message_pool["round_id"] == 0:
            self.proto_grad = self.encode_gradient(task, task_data)
        else:
            self.proto_grad = None

        with torch.no_grad():
            for (local_param_old, agg_global_param) in zip(self.local_model.parameters(), self.message_pool["server"]["weight"]):
                local_param_old.data.copy_(agg_global_param)
        with torch.no_grad():
            for (local_param_old, agg_global_param) in zip(self.global_model.parameters(), self.message_pool["server"]["weight"]):
                local_param_old.data.copy_(agg_global_param)
        
        self.local_model.train()

        for epoch_i in range(self.args.num_epochs):
            self.optim.zero_grad()
            logits, embedding, _ = self.local_model.forward(task_data)
            loss_train = self.loss_fn(logits[task["train_mask"]], whole_data.y[task["train_mask"]])
            
            if self.local_buffer["x"] is not None:
                buffer_data = isolate_graph(x=self.local_buffer["x"], y=self.local_buffer["y"])
                buffer_logits, buffer_embedding, _ = self.local_model.forward(buffer_data)
                loss_replay = self.loss_fn(buffer_logits, buffer_data.y)
                loss_train = loss_train * self.args.beta + loss_replay * (1-self.args.beta)
            
            loss_train.backward()
            self.optim.step()

        self.local_model.eval()
        
        
        
    def task_done(self, task_id):
        whole_data = self.data["data"].to(self.device)
        task = self.data["task"][task_id]
        
        if task_id not in self.cache:
            task_data = self.task_data(task_id, whole_data, task)
            self.cache[task_id] = task_data
            
        task_data = self.cache[task_id]
        
        self.local_model.eval()
        _, embedding_local, _ = self.local_model.forward(task_data)
        _, embedding_global, _ = self.global_model.forward(task_data)
        embedding = self.args.alpha * embedding_local + (1-self.args.alpha) * embedding_global
        
        
        self.local_buffer = update_buffer(buffer=self.local_buffer, replay=self.args.replay, task=task, task_data=task_data, embedding=embedding, num_samples_per_class=self.args.num_samples_per_class)

    def send_message(self, task_id):
        self.message_pool[f"client_{self.client_id}"] = {
                "num_samples": self.get_task_num_samples(task_id),
                "weight": list(self.local_model.parameters()),
                "proto_grad": self.proto_grad 
            }
    
    def evaluate(self, task_id, use_global=False, mask="test_mask"):
        if use_global:
            local_param_copy = copy.deepcopy(list(self.local_model.parameters()))
            with torch.no_grad():
                for (local_param, global_param) in zip(self.local_model.parameters(), self.message_pool["server"]["weight"]):
                    local_param.data.copy_(global_param)
            
        self.local_model.eval()
        whole_data = self.data["data"].to(self.device)
        task = self.data["task"][task_id]
        
        if task_id not in self.cache:
            task_data = self.task_data(task_id, whole_data, task)
            self.cache[task_id] = task_data
            
        task_data = self.cache[task_id]
        
        logits, embedding, _ = self.local_model.forward(task_data)
        acc_task_test = accuracy(logits[task[mask]], whole_data.y[task[mask]])
        
        if use_global:
            with torch.no_grad():
                for (local_param, global_param) in zip(self.local_model.parameters(), local_param_copy):
                    local_param.data.copy_(global_param)
        
        return acc_task_test
        
    def task_data(self, task_id, whole_data, task):
        handled = task["train_mask"] | task["val_mask"] | task["test_mask"]
        masked_edge_index_filename = os.path.join(self.args.task_dir, f"client_{self.client_id}_task_{task_id}.pt")
        if not os.path.exists(masked_edge_index_filename):
            masked_edge_index = edge_masking(whole_data.edge_index, handled=handled, device=self.device)
            torch.save(masked_edge_index, masked_edge_index_filename)
        else:
            masked_edge_index = torch.load(masked_edge_index_filename, map_location=self.device)
            
        task_data = Data(x=whole_data.x, edge_index=masked_edge_index, y=whole_data.y)
        return task_data

            
        
    def encode_gradient(self, task, task_data):
        self.ge_model.train()
        proto_grad = []
        selected_nodes = task["train_mask"]
        ground_truth = task_data.y[selected_nodes]
        current_classes = torch.unique(ground_truth).tolist()
        for class_i in current_classes:
            class_i_prototype = torch.mean(task_data.x[selected_nodes][ground_truth == class_i], dim=0)
            num_class_i = (ground_truth == class_i).sum()
            print(f"[client {self.client_id} task {self.message_pool['task_id']} round {self.message_pool['round_id']}]\tclass: {class_i}\ttotal_nodes: {num_class_i}")
    
            outputs = self.ge_model.forward(class_i_prototype)
            loss_cls = nn.CrossEntropyLoss()(outputs, torch.tensor(class_i).long().to(self.device))
            dy_dx = torch.autograd.grad(loss_cls, self.ge_model.parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))
            proto_grad.append((original_dy_dx, num_class_i))
        return proto_grad
        
        
        
        
class OursServer(BaseServer):
    def __init__(self, args, message_pool, device):
        super(OursServer, self).__init__(args, message_pool, device)
        self.local_models = [load_model(name=args.model, input_dim=args.input_dim, hid_dim=args.hid_dim, output_dim=args.output_dim, dropout=args.dropout).to(self.device) for _ in range(self.args.num_clients)]
        self.global_model = load_model(name=args.model, input_dim=args.input_dim, hid_dim=args.hid_dim, output_dim=args.output_dim, dropout=args.dropout).to(self.device)
        self.ge_model = GEModel(input_dim=args.input_dim, output_dim=args.output_dim).to(device)
        self.ge_message = False
        self.each_class_buffers = {class_i:[] for class_i in range(self.args.output_dim)}
        self.kd_weight = []
        
    def execute(self):
        with torch.no_grad():
            num_tot_samples = sum([self.message_pool[f"client_{client_id}"]["num_samples"] for client_id in range(self.args.num_clients)])
            for task_id, client_id in enumerate(range(self.args.num_clients)):
                weight = self.message_pool[f"client_{client_id}"]["num_samples"] / num_tot_samples
                for (local_param, global_param) in zip(self.message_pool[f"client_{client_id}"]["weight"], self.global_model.parameters()):
                    if task_id == 0:
                        global_param.data.copy_(weight * local_param)
                    else:
                        global_param.data += weight * local_param
                        
        
        if self.message_pool["round_id"] == 0:
            self.kd_weight.append(torch.zeros(size=(self.args.num_clients, self.args.output_dim)).to(self.device))
            for client_id in range(self.args.num_clients):
                for (grad, num) in self.message_pool[f"client_{client_id}"]["proto_grad"]:
                    label = self.gradient2label(grad)
                    self.kd_weight[self.message_pool["task_id"]][client_id, label] += num
                    dummy_x = torch.randn(self.args.input_dim).unsqueeze(0).to(self.device).requires_grad_(True)
                    label_pred = torch.Tensor([label]).long().to(self.device).requires_grad_(False)
                    optimizer = torch.optim.LBFGS([dummy_x, ], lr=self.args.LBFGS_init_lr)
                    criterion = nn.CrossEntropyLoss().to(self.device)

                    recon_model = copy.deepcopy(self.ge_model).to(self.device)

                    for iters in range(self.args.num_it_recon):
                        def closure():
                            optimizer.zero_grad()
                            pred = recon_model.forward(dummy_x)
                            dummy_loss = criterion(pred, label_pred)

                            dummy_dy_dx = torch.autograd.grad(dummy_loss, recon_model.parameters(), create_graph=True)

                            grad_diff = 0
                            for gx, gy in zip(dummy_dy_dx, grad):
                                grad_diff += ((gx - gy) ** 2).sum()
                            grad_diff.backward()
                            return grad_diff

                        optimizer.step(closure)
                        current_loss = closure().item()

                        if iters == self.args.num_it_recon - 1:
                            print(f"current_loss:{current_loss}")

                        if iters >= self.args.num_it_recon - self.args.num_recon_nodes:
                            dummy_data_temp = dummy_x.requires_grad_(False).clone()
                            self.each_class_buffers[label].append(dummy_data_temp)
            
                    print(f"[server] reconstruct {self.args.num_recon_nodes} nodes with class {label} based on client {client_id}'s uploaded prototype gradient.")

        for client_id in range(self.args.num_clients):
            with torch.no_grad():
                for (old_local_param, local_param) in zip(self.local_models[client_id].parameters(), self.message_pool[f"client_{client_id}"]["weight"]):
                        old_local_param.data.copy_(local_param)
            self.local_models[client_id].eval()
            
        optimizer_g = Adam(self.global_model.parameters(), lr=self.args.lr_g, weight_decay=self.args.weight_decay)
        self.global_model.train()
        

        
        normalized_kd_weight = copy.deepcopy(self.kd_weight)


        for task_id in range(self.message_pool["task_id"]+1):
            for it, prev_task_id in enumerate(range(0, task_id)):
                normalized_kd_weight[task_id] += self.kd_weight[prev_task_id] * (self.args.decay ** (task_id-it))
              



        for task_id in range(self.message_pool["task_id"]+1):
            for class_i in range(self.args.output_dim):
                if normalized_kd_weight[task_id][:, class_i].sum() != 0:
                    normalized_kd_weight[task_id][:, class_i] /= normalized_kd_weight[task_id][:, class_i].sum()
            
                    
                    
                    
        x_all = []
        each_class_mask = {}
        for class_i in range(self.args.output_dim):
            each_class_mask[class_i] = list(range(len(x_all), len(x_all)+len(self.each_class_buffers[class_i])))
            x_all += self.each_class_buffers[class_i]
        
        buffer_graph = construct_self_loop_graph(x=torch.vstack(x_all))
        for class_i in range(self.args.output_dim):
            each_class_mask[class_i] = idx_to_mask_tensor(each_class_mask[class_i], len(x_all)).to(self.device).bool()
            
                    
                    
        for epoch_g in range(self.args.num_epoch_g):
            optimizer_g.zero_grad()
            loss_fgt = 0
            
            global_logits, _, _ = self.global_model.forward(buffer_graph)
            
            for client_id in range(self.args.num_clients):
                local_logits, _, _ = self.local_models[client_id].forward(buffer_graph)
                
                for class_i in range(self.args.output_dim):
                    if len(self.each_class_buffers[class_i]) == 0:
                        continue
                    loss_fgt += normalized_kd_weight[self.message_pool["task_id"]][client_id, class_i] * \
                        torch.mean(torch.mean(torch.abs(global_logits[each_class_mask[class_i]] - local_logits[each_class_mask[class_i]]), dim=1))                    
               
            loss_fgt.backward()
            optimizer_g.step()

        
        self.global_model.eval()

           
    def gradient2label(self, grad):
        pred = torch.argmin(grad[-1]).detach().reshape((1,)).requires_grad_(False)
        return pred.item()
    
     
    def send_message(self):
        self.message_pool["server"] = {
            "weight": list(self.global_model.parameters())
        }
        if not self.ge_message:
            self.message_pool["server"]["ge"] = list(self.ge_model.parameters())
            self.ge_message = True
        
