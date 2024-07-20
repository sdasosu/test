class Statistics:
    def __init__(self, model, data_width, train_flops, valid_flops) -> None:
        self.model = model
        self.data_width = data_width
        self.train_flops = train_flops
        self.valid_flops = valid_flops
        
        #----------- Model Size units -------
        Byte = 8
        KiB = 1024 * Byte
        MiB = 1024 * KiB
        GiB = 1024 * MiB

        #-----------Model macs ---------------
        total_param_count = 0
        non_zero_param_count = 0
        sparsity=0

        for param in model.parameters():
            total_param_count += param.numel()
            non_zero_param_count += param.count_nonzero().item()

        unsparsed_model_size = (total_param_count * data_width) / MiB
        sparsed_model_size = (non_zero_param_count * data_width) / MiB
        sparsity= (1.0 - (non_zero_param_count/total_param_count))

        print("\n\n------------- Printing Model Statistics -------------------\n")
        print(f"\n Unsparsed Model Size         = {unsparsed_model_size:.2f} MiB")
        print(f"\n Sparsity (zero values)       = {sparsity:.2f}")
        print(f"\n Sparsed (Pruned) Model Size  = {sparsed_model_size:.2f} MiB")
        print(f"\n ----------------------------------------------------------\n\n")
        print(f"\n Training Flops count         : {self.train_flops} GFLOP")
        print(f"\n Total Training Flops         : {sum(self.train_flops) / 1e9:.2f} GFLOP")
        print(f"\n Validation Flops count       : {self.valid_flops} GFLOP")
        print(f"\n Total Validation Flops       : {sum(self.valid_flops) / 1e9:.2f} GFLOP")
        print("\n---------------------------------------------------------------\n\n")

