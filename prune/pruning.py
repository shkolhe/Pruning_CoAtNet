import os
import copy
import torch
import torch.nn.utils.prune as prune
from utils.utils import *

def remove_parameters(model):

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model


def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):

    total_zeros = 0
    total_elements = 0

    if (use_mask == True):
        for buffer_name, buffer in module.named_buffers():
            if ("weight_mask" in buffer_name) and (weight == True):
                total_zeros += torch.sum(buffer == 0).item()
                total_elements += buffer.nelement()
            if ("bias_mask" in buffer_name) and (bias == True):
                total_zeros += torch.sum(buffer == 0).item()
                total_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if ("weight" in param_name) and (weight == True):
                total_zeros += torch.sum(param == 0).item()
                total_elements += param.nelement()
            if ("bias" in param_name) and (bias == True):
                total_zeros += torch.sum(param == 0).item()
                total_elements += param.nelement()

    sparsity = total_zeros / total_elements    

    return total_zeros, total_elements, sparsity


def measure_global_sparsity(model, weight=True, bias=False,
                            conv2d_use_mask=False, linear_use_mask=False):
    
    total_zeros = 0
    total_elements = 0

    for module_name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):

            module_zeros, module_elements, _ = measure_module_sparsity(
                module, weight, bias, use_mask=conv2d_use_mask)
            total_zeros += module_zeros
            total_elements += module_elements

        elif isinstance(module, torch.nn.Linear):

            module_zeros, module_elements, _ = measure_module_sparsity(
                module, weight, bias, use_mask=linear_use_mask)
            total_zeros += module_zeros
            total_elements += module_elements
    
    sparsity = total_zeros / total_elements    

    return total_zeros, total_elements, sparsity


def iterative_pruning(model, train_loader, test_loader,
                      learning_rate, device, model_name,
                      decay=0.1, conv2d_prune_perc=0.4, 
                      linear_prune_perc=0.2, iterations=10,
                      epochs_per_iteration=10, model_prefix="pruned", 
                      model_dir="saved_images", global_pruning=False):
    
    for i in range(iterations):

        print("Pruning {}/{}".format(i+1, iterations))
        print("Starting Pruning...")

        if (global_pruning == True):

            params_to_prune = []
            for module_name, module in model.named_modules():

                if isinstance(module, torch.nn.Conv2d):
                    params_to_prune.append((module, "weight"))
            
            prune.global_unstructured(params_to_prune,
                                      pruning_method=prune.L1Unstructured,
                                      amount=conv2d_prune_perc,)
        
        else:

            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module, name="weight", amount=conv2d_prune_perc)
                elif isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module, name="weight", amount=linear_prune_perc)
        
        _, eval_acc = evaluate_model(model, test_loader, device, None)

        total_zeros, total_elements, sparsity = measure_global_sparsity(model, weight=True,
                                   bias=False, conv2d_use_mask=True, linear_use_mask=False)
        
        print("Test Accuracy: {:.3f}".format(eval_acc))
        print("Global Sparsity: {:.2f}".format(sparsity))

        print("Starting Fine-tuning...")

        new_lr = learning_rate * (decay**i)
        train_model(model, train_loader, test_loader, new_lr, decay, epochs_per_iteration, device)

        _, eval_acc = evaluate_model(model, test_loader, device, None)

        total_zeros, total_elements, sparsity = measure_global_sparsity(model, weight=True,
                                   bias=False, conv2d_use_mask=True, linear_use_mask=False)

        print("Test Accuracy: {:.3f}".format(eval_acc))
        print("Global Sparsity: {:.2f}".format(sparsity))

        # Save Model
        model_filename = "{}_{}_{}.h5".format(model_prefix, model_name[:-3], i+1)
        save_model(model, model_filename, model_dir)

        # Load and Return the Pruned model
        pruned_model = load_model(model_filename, model_dir, device)

    return pruned_model

