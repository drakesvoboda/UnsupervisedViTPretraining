import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCE(nn.Module):
    """ We term it NT-Xent (the normalized temperature-scaled cross entropy loss) - Quote from paper """

    def __init__(self, tempurature=0.07):
        super().__init__()
        self.tempurature = tempurature
        self.cos = nn.CosineSimilarity(n=2)

        pass

    def forward(self, z_i, z_j):
        """ Compute loss for positive pairs z_i and z_j.          

                We do not sample negative examples explicitly. Instead, given a positive pair, 
                similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples. - Quote from paper
        """


        z = torch.cat((z_i, z_j), dim=0)

        dists = self.cos(z.unsqueeze(1), z.unsqueeze(0)) / self.tempurature

        pass




