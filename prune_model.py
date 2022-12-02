import os
import torch
from model.coatnet import *
from prune.pruning import *
from utils.utils import *

def main():

    random_seed = 3
    num_classes = 10
    lr = 1e-3
    decay = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_random_seeds(random_seed)
    
    # Save the trained model under saved_images directory
    model_dir = "./saved_images"
    model_prefix = "pruned"
    model_name = "cifar10_coatnet0.h5"

    # Load the Pre-Trained model
    model = load_model(model_name, model_dir, device)

    train_loader, test_loader, classes = prepare_dataloader(num_workers=8,
                                         train_batch_size=128, eval_batch_size=256)
    
    _, eval_acc = evaluate_model(model, test_loader, device, None)
    print("Test Accuracy: {:.3f}".format(eval_acc))

    total_zeros, total_elements, sparsity = measure_global_sparsity(model)
    print("Global Sparsity: {:.2f}".format(sparsity))

    print("Starting Iterative Pruning...")

    pruned_model = copy.deepcopy(model)

    iterative_pruning(pruned_model, train_loader, test_loader, lr, device,
                      model_name, decay, conv2d_prune_perc=0.98, linear_prune_perc=0,
                      iterations=1, epochs_per_iteration=200, model_prefix=model_prefix,
                      model_dir=model_dir, global_pruning=True)
    
    # Apply the Pruning and remove the mask/parameters
    remove_parameters(pruned_model)

    # Earlier model
    print("Original model")
    _, eval_acc = evaluate_model(model, test_loader, device, None)
    print("Test Accuracy: {:.3f}".format(eval_acc))

    total_zeros, total_elements, sparsity = measure_global_sparsity(model)
    print("Global Sparsity: {:.2f}".format(sparsity))

    # Pruned model
    print("Pruned model")
    _, eval_acc = evaluate_model(pruned_model, test_loader, device, None)
    print("Test Accuracy: {:.3f}".format(eval_acc))

    total_zeros, total_elements, sparsity = measure_global_sparsity(pruned_model)
    print("Global Sparsity: {:.2f}".format(sparsity))


if __name__ == "__main__":

    main()

