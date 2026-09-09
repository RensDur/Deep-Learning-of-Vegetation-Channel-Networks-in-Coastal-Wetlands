import os
import torch
import matplotlib.pyplot as plt
from spline.kernels import *





def first_order():
    
    xs = torch.linspace(-1, 1, 51)
    
    fig, ax = plt.subplots()

    fig.tight_layout(pad=1.5)
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    
    ax.plot(xs, p1_1(xs), label=r"$h^0_0(x)$")
    
    ax.axhline(0, color='black', linewidth=0.8)  # horizontal line at y=0
    ax.axvline(0, color='black', linewidth=0.8)  # vertical line at x=0

    ax.set(title=r"Basis splines for $n=0$", xlabel="x", ylabel="y")
    ax.legend()

    os.makedirs(f"./Hermite spline basis splines", exist_ok=True)
    fig.savefig(f"./Hermite spline basis splines/First order basis splines.jpg", dpi=150)
    
    # plt.show()


def second_order():
    
    xs = torch.linspace(-1, 1, 51)
    
    fig, ax = plt.subplots()
    
    fig.tight_layout(pad=1.5)
        
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    
    ax.plot(xs, p2_1(xs), label=r"$h^1_0(x)$")
    ax.plot(xs, p2_2(xs), label=r"$h^1_1(x)$")
    
    ax.axhline(0, color='black', linewidth=0.8)  # horizontal line at y=0
    ax.axvline(0, color='black', linewidth=0.8)  # vertical line at x=0

    ax.set(title=r"Basis splines for $n=1$", xlabel="x", ylabel="y")
    ax.legend()

    os.makedirs(f"./Hermite spline basis splines", exist_ok=True)
    fig.savefig(f"./Hermite spline basis splines/Second order basis splines.jpg", dpi=150)
    
    # plt.show()


def third_order():
    
    xs = torch.linspace(-1, 1, 51)
    
    fig, ax = plt.subplots()

    fig.tight_layout(pad=1.5)
    
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    
    ax.plot(xs, p3_1(xs), label=r"$h^2_0(x)$")
    ax.plot(xs, p3_2(xs), label=r"$h^2_1(x)$")
    ax.plot(xs, p3_3(xs), label=r"$h^2_2(x)$")
    
    ax.axhline(0, color='black', linewidth=0.8)  # horizontal line at y=0
    ax.axvline(0, color='black', linewidth=0.8)  # vertical line at x=0

    ax.set(title=r"Basis splines for $n=2$", xlabel="x", ylabel="y")
    ax.legend()

    os.makedirs(f"./Hermite spline basis splines", exist_ok=True)
    fig.savefig(f"./Hermite spline basis splines/Third order basis splines.jpg", dpi=150)
    
    # plt.show()


def main():

    first_order()
    second_order()
    third_order()





if __name__ == "__main__":
    main()