import torch
import torch.nn as nn


class Scale(nn.Module):
    r"""Scales the input vector by a given scalar.
    """

    def __init__(self, factor):
        super(Scale, self).__init__()
        self.factor = factor

    def reset_parameters(self):
        pass

    def forward(self, input):
        factor = torch.Tensor([self.factor]).to(input.device)
        if self.factor == 1:  # This is to make sure this operation is not backpropped on, or unnecessarily computed.
            return input
        else:
            return factor * input

    def extra_repr(self):
        return 'factor={}'.format(self.factor)
