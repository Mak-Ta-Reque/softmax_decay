import sys

from datetime import datetime
import os, sys
import torchvision
from utils import load_expl_dir_form, append_evaluation_result, get_missing_run_parameters, update_eval_result, load_expl, arg_parse
import json
import time
from torchvision import models
## import from road module
import road
from road import run_road
from road.imputations import *
from road.retraining import *
from experiments.explanation_generation.utils import Data_Loader
# different seeds
seeds = [2005, 42, 1515, 3333, 420]

if __name__ == '__main__':
    ## read configs
    args = arg_parse()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_path = args.data_path
    expl_path = args.expl_path
    use_device = torch.device("cuda:0" if args.gpu else "cpu")
    batch_size = args.batch_size
    params_file = args.params_file
    dataset = args.dataset
    model_path = args.model_path
    image_size = (224, 224)

   


    ## set transforms
    transform_train = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]) #transforms.RandomHorizontalFlip(),
    transform_test = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    # Apply this transformation after imputation.
    normalize_transform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transform_tensor = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    
    params = json.load(open(params_file))
    print("Device: ", use_device)
    print("Base Method Group: ", params["basemethod"])
    print("Types:", params["modifiers"])
    print("Imputation: ", params["imputation"])
    print("MoRF-order", bool(params["morf"]))
    print("Ranking: ", params["ranking"])
    print("Resultsfile",  params["datafile"])
    print("Percentages", params["percentages"])
    print("Timeout", int(params["timeoutdays"]))
    imputation = params["imputation"]
    group = params["basemethod"]
    morf = bool(params["morf"])
    storage_file = params["datafile"]
    modifiers = params["modifiers"]
    ps = params["percentages"]
    ranking = params["ranking"]


    
    if imputation == "linear":
        imputer = NoisyLinearImputer(noise=0.01)
    elif imputation == "telea":
        imputer = ImpaintingImputation()
    elif imputation == "ns":
        imputer = ImpaintingImputationNS()
    elif imputation == "fixed":
        imputer = ChannelMeanImputer()
    elif imputation == "gain":
        imputer = GAINImputer("../../road/gisp/models/cifar_10_best.pt", use_device)
    elif imputation == "zero":
        imputer = ZeroImputer()

    # Load trained model
    
    run_params = get_missing_run_parameters(storage_file, imputation, morf, group, modifiers, ps, timeout=int(params["timeoutdays"]))
    print("Got Run Parameters (mod, perc, run_id): ", run_params)
 

    model = None
    dataset_test = None
    expl_test = None
    if dataset=="cifer10":
            num_of_classes = 10
            model = models.resnet18()
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_of_classes)
            # load trained classifier
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            dataset_test= torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_tensor)
            # to do enble cifer 10
           
    
    elif dataset == "food101":
            num_of_classes = 101
            model = models.resnet50()
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_of_classes)
            model.load_state_dict(torch.load(model_path, map_location=torch.device(use_device))["model_state"])
            dataset_test = Data_Loader(root=data_path, train=False, dataset='Food-101', transform=transform_tensor)
            # list all the files from input dir
            

    while run_params is not None:
        modifier = run_params[0]
        perc_value = run_params[1]
        run_id = run_params[2]
        torch.manual_seed(seeds[run_id]) # set appropriate seed 
        
        if dataset=="cifer10":
            expl_train = f"{expl_path}/{group}/{modifier}_train.pkl"
            expl_test = f"{expl_path}/{group}/{modifier}_test.pkl"
            _, expl_test, _, pred_test = load_expl(None, expl_test)

        
        elif dataset == "food101":
            expl_train = f"{expl_path}/{group}_{modifier}"
            expl_test = f"{expl_path}/{group}_{modifier}"
            expl_test, pred_test = load_expl_dir_form(expl_test, train=False) # return a indexing object like list

        # load trained classifier

        start_time = time.time()
        ## load cifar 10 dataset in tensors
        #transform_tensor = transforms.Compose([transforms.ToTensor()])
        #cifar_train = torchvision.datasets.CIFAR10(root=data_local, train=True, download=True, transform=transform_tensor)
        #dataset_test = Data_Loader(root=data_path, train=False, dataset='Food-101', transform=transform_test)

        ## load explanation
        #_, explanation_train, _, prediction_train = load_expl(None, expl_train)

        res_acc, prob_acc = run_road(model, dataset_test, expl_test, normalize_transform, [perc_value], morf=morf, batch_size=32, imputation = imputer, ranking=ranking)
        # res_acc, prob_acc = run_road(model, dataset_test, expl_test, normalize_transform, [perc_value], morf=morf, batch_size=32, imputation = imputer)
        print('finished job with params', run_params, " Drawing new params.")
        print('--' * 50)
        print("--- %s seconds ---" % (time.time() - start_time))

        update_eval_result(res_acc[0].item(), storage_file, imputation, group, modifier, morf, perc_value, run_id)
        run_params = get_missing_run_parameters(storage_file, imputation, morf, group, modifiers, ps)
        print("Got Run Parameters (mod, perc, run_id): ", run_params)
            # exit()

    print("No more open runs. Terminiating.")
