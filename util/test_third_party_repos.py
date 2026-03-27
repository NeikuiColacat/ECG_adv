import torch
import time

def main() :
    while True : 
        a = torch.randn(1025,1025, 1025 , device="cuda")
        b = torch.randn(1025,1025, 1025 , device="cuda")

        c = a @ b

        print(c.mean().item())



if __name__ == "__main__" :
    main()

