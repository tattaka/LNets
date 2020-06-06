import torch

from lnets.utils.math.projections import *


def project_weights(weight, proj_config):
    with torch.no_grad():
        if proj_config.type == "l_2":
            # scaling = get_safe_bjorck_scaling(weight, cuda=cuda)
            projected_weights = project_on_l2_ball(weight.t(),
                                                   bjorck_iter=proj_config.bjorck_iter,
                                                   bjorck_order=proj_config.bjorck_order,
                                                   bjorck_beta=proj_config.bjorck_beta).t()

        elif proj_config.type == "l_inf_projected":
            projected_weights = project_on_linf_ball(weight)

        elif proj_config.type == "l_inf_scaled":
            projected_weights = scale_on_linf_ball(weight,
                                                   scale_all=proj_config.scale_all)

        else:
            print("Requested projection type not recognized. ")
            exit(-1)

        return projected_weights
