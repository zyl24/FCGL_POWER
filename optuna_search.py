
from config import args
import torch
import copy
from data import load_fcgl_dataset
from utils import load_clients_server, start, seed_everything
import optuna

    


first_initialized_copy = {"flag": False, "clients": None, "server": None}



def objective(trial: optuna.Trial):
    args.num_recon_nodes = trial.suggest_categorical(name="num_recon_nodes", choices=[1,3])
    args.num_it_recon = 300
    args.beta = trial.suggest_float(name="beta", low=0.01, high=0.1, step=0.01)
    args.decay = trial.suggest_float(name="decay", low=0, high=1, step=0.1)
    args.num_epoch_g = trial.suggest_categorical(name="num_epoch_g", choices=[3, 5, 10])
    
    # load clients, server, message_pool
    clients, server, message_pool = load_clients_server(args, fcgl_dataset, device)
    
    if first_initialized_copy["flag"]:
        # set parameters
        for client_id in range(args.num_clients):
            with torch.no_grad():
                for (local_param_old, initialized_param) in zip(clients[client_id].local_model.parameters(), first_initialized_copy["clients"][client_id].local_model.parameters()):
                    local_param_old.data.copy_(initialized_param)
            with torch.no_grad():
                for (local_param_old, initialized_param) in zip(clients[client_id].ge_model.parameters(), first_initialized_copy["clients"][client_id].ge_model.parameters()):
                    local_param_old.data.copy_(initialized_param)

        with torch.no_grad():
            for (global_param_old, initialized_param) in zip(server.global_model.parameters(), first_initialized_copy["server"].global_model.parameters()):
                global_param_old.data.copy_(initialized_param)
        with torch.no_grad():
            for (global_param_old, initialized_param) in zip(server.ge_model.parameters(), first_initialized_copy["server"].ge_model.parameters()):
                global_param_old.data.copy_(initialized_param)
    else:
        first_initialized_copy["flag"] = True
        first_initialized_copy["clients"] = copy.deepcopy(clients)
        first_initialized_copy["server"] = copy.deepcopy(server)
    
    
    return start(args, fcgl_dataset, clients, server, message_pool, device)[0]


if __name__ == "__main__":   
    args.disable_cuda = True

    seed_everything(args.seed)
    if not args.disable_cuda:
        device = torch.device(f"cuda:{args.gpuid}")
    else:
        device = torch.device(f"cpu")
        
    fcgl_dataset, input_dim, output_dim, task_dir = load_fcgl_dataset(root=args.root, dataset=args.dataset, num_clients=args.num_clients, classes_per_task=args.num_classes_per_task, shuffle_task=args.shuffle_task)
    args.input_dim = input_dim
    args.output_dim = output_dim
    args.task_dir = task_dir

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=300)

    print('Best trial:')
    trial = study.best_trial
    print('  Best Global AA: {}'.format(trial.value))
    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    completed_trials = study.get_trials(deepcopy=True, states=[optuna.trial.TrialState.COMPLETE])

    sorted_trials = sorted(completed_trials, key=lambda trial: trial.value, reverse=True)

    for trial in sorted_trials:
        print(f"Trial {trial.number}: Value {trial.value}, Params {trial.params}")
