import os
import torch
from model.coatnet import *
from utils.utils import *

def main():

    random_seed = 3
    num_classes = 10
    lr = 1e-3
    decay = 1e-5
    num_epochs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    set_random_seeds(random_seed)
    
    # Save the trained model under saved_images directory
    model_dir = "./saved_images"
    model_name = "cifar10_coatnet0.h5"

    # Load the untrained model
    #model = load_model(model_name, model_dir, device)
    model = coatnet_0(num_classes)

    # Create Train and Test DataLoaders
    train_loader, test_loader, classes = prepare_dataloader(
        num_workers=8, train_batch_size=128, eval_batch_size=256)
    
    # Train model
    print("Training Model")
    model = train_model(model, train_loader,
                        test_loader, decay,
                        lr, num_epochs, device)
    
    # Save model
    save_model(model, model_name, model_dir)

    # Load the trained model
    model = load_model(model_name, model_dir, device)
    # Evaluate
    _, eval_acc = evaluate_model(model, test_loader, device, None)
    print("Test Accuracy: {:.3f}".format(eval_acc))


if __name__== "__main__":

    main()