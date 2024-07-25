    import os
    import torch

    from smallNet import *
    from torch import optim
    from utils import test, load_data


    def main():
        result_folder_name = 'Results'
        weights_file_name = "mnist_cnn_cpu.pt"
        os.makedirs(result_folder_name, exist_ok=True)

        batch_size = 64
        epochs = 10
        lr = 0.01
        momentum = 0.5
        seed = 1

        torch.manual_seed(seed)
        use_cuda = False  # torch.cuda.is_available()

        device = torch.device("cpu")  # torch.device("cuda" if use_cuda else "cpu") 

        # call load_data
        train_loader, test_loader = load_data(batch_size, use_cuda)

        model = SmallNet().to(device)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        for _ in range(epochs):
            for source, target in train_loader:
                source = source.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(source)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
        test(model, device, test_loader)

        torch.save(model.state_dict(), os.path.join(
            result_folder_name, weights_file_name))

        return 0


    if __name__ == '__main__':
        main()
