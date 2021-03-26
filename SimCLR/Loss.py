import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCE(nn.Module):
    """ We term it NT-Xent (the normalized temperature-scaled cross entropy loss) - Quote from paper """

    def __init__(self, tempurature=0.07):
        super().__init__()
        self.tempurature = tempurature
        self.cos = nn.CosineSimilarity(dim=2)
        self.bs = -1 
        self.neg_mask = None

    def forward(self, embeds: Tuple[torch.tensor, torch.tensor], *args):
        """ Compute loss for positive pairs z_i and z_j.          

            We do not sample negative examples explicitly. Instead, given a positive pair, 
            similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples. - Quote from paper

            Computes matrix pairwise cosine similarity between vecotrs in z_i and z_j. 
            Below is an example with a batch size of 3. The is and js are embeddings of the same image with different augmentations:

               | i1 | i2 | i3 | j1 | j2 | j3 |
            ---+----+----+----+----+----+----+
            i1 | 1  | n  | n  | PP | n  | n  | 
            ---+----+----+----+----+----+----+
            i2 | n  | 1  | n  | n  | PP | n  | 
            ---+----+----+----+----+----+----+
            i3 | n  | n  | 1  | n  | n  | PP |
            ---+----+----+----+----+----+----+
            j1 | PP | n  | n  | 1  | n  | n  |
            ---+----+----+----+----+----+----+
            j2 | n  | PP | n  | n  | 1  | n  |
            ---+----+----+----+----+----+----+
            j3 | n  | n  | PP | n  | n  | 1  |
            ---+----+----+----+----+----+----+

            Cells with `PP` denote similarities between positive pairs, cells with `n` denote negative pairs.
            Cells with 1 are discarded.
        """

        z_i, z_j = embeds

        bs, h = z_i.shape

        z = torch.cat((z_i, z_j), dim=0)
                
        sims = self.cos(z.unsqueeze(1), z.unsqueeze(0)) / self.tempurature # Compute pairwise similarity matrix

        posi = torch.diag(sims, bs)  # Get positive pair similarities of top right quadrant
        posj = torch.diag(sims, -bs) # Get positive pair similarities of bottom left quadrant

        pos = torch.cat((posi, posj), dim=0).unsqueeze(1)

        if bs != self.bs:
            self.bs = bs
            self.neg_mask = torch.ones((bs*2, bs*2), dtype=bool).fill_diagonal_(0)
            torch.diagonal(self.neg_mask, bs).fill_(0)
            torch.diagonal(self.neg_mask, -bs).fill_(0)

        neg = sims[self.neg_mask].reshape(bs*2, -1)

        logits = torch.cat((pos, neg), dim=1)

        return F.cross_entropy(logits, torch.zeros(bs*2, dtype=torch.long).to(logits.device))



