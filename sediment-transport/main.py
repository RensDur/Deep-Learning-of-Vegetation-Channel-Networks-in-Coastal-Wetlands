import torch
from window import Window
from numerical import Solver


def main():

    width = 200
    height = 200

    torch_device = torch.device("cpu")

    if torch.backends.mps.is_available():
        torch_device = torch.device("mps")

    print(f"Using torch device: {torch_device}")
    
    win = Window("h", width, height)
    win.set_data_range(0, 1)

    solver = Solver(width, height, torch_device)



if __name__ == "__main__":
    main()