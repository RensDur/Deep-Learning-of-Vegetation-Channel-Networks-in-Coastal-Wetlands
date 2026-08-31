import torch




def main():

    characteristic_scales_old = torch.load(f"./snapshots-log-slowdown/characteristic_scales_per_sample_old.pt").unsqueeze(2).repeat(1, 1, 100)
    characteristic_scales = torch.load(f"./snapshots-log-slowdown/characteristic_scales_per_sample.pt").unsqueeze(2).repeat(1, 1, 100)

    # Load the residuals from disk and merge them into one tensor
    evaluation_residuals = torch.zeros(500, 5, 100) # Number of samples per ablation, number of channels, sample every 10 iterations for 1000 iters

    for i in range(0, 500, 10):
        evaluation_residuals[i:(i+10)] = torch.load(f"./Hybrid Hydro-PINN evaluation/eval_residuals/sfere_start {i} sfere_end {i+10}.pt")

    # Allocate memory for rescaled evaluation residuals
    rescaled_evaluation_residuals = torch.zeros_like(evaluation_residuals)


    # Rescaling of residuals is done by multiplying by the old scale and normalising wrt the new ones
    rescaled_evaluation_residuals[:, 0, :] = evaluation_residuals[:, 0, :] * characteristic_scales_old[:, 0, :] / characteristic_scales[:, 0, :]
    rescaled_evaluation_residuals[:, 1, :] = evaluation_residuals[:, 1, :] * characteristic_scales_old[:, 1, :] / characteristic_scales[:, 1, :]
    rescaled_evaluation_residuals[:, 2, :] = evaluation_residuals[:, 2, :] * characteristic_scales_old[:, 2, :] / characteristic_scales[:, 2, :]
    rescaled_evaluation_residuals[:, 3, :] = evaluation_residuals[:, 3, :] * characteristic_scales_old[:, 3, :] / characteristic_scales[:, 3, :]
    rescaled_evaluation_residuals[:, 4, :] = evaluation_residuals[:, 4, :] * characteristic_scales_old[:, 4, :] / characteristic_scales[:, 4, :]

    # Save the rescaled residuals
    for i in range(0, 500, 10):
        torch.save(rescaled_evaluation_residuals[i:(i+10)], f"./Hybrid Hydro-PINN evaluation/eval_residuals/sfere_start {i} sfere_end {i+10}.pt")



if __name__ == "__main__":
    main()