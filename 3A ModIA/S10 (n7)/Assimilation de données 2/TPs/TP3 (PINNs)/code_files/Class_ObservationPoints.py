import torch

class ObservationPoints:
    """
    ObservationPoints class, used to generate the observations given the reference solution. 
    """
    
    def __init__(self, ref_solution, N_obs = 10, noise_std = 0, 
                 seed = None):
        
        # Seed for random obvservations reproductibility
        if (seed != None):
            torch.manual_seed(seed)
        
        self.N_obs = N_obs
        self.regime = ref_solution['regime']
        
        N = ref_solution['domain'].shape[0]
    
        self.indices = torch.randperm(N)[:N_obs].detach().clone()
        self.data = torch.hstack((ref_solution['domain'], 
                                  ref_solution['solution'].detach().clone()+torch.randn(N, 1)*noise_std))[self.indices].detach().clone()