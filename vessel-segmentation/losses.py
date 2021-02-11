import sys

import numpy as np

from dataclasses import dataclass


@dataclass
class HParams:
    lambda1: float
    lambda2: float
    lambda3: float


def get_profile(image, vessel_flow, vessel_radius):
    # the profile of image I at point p along the direction u.
    return 0


def get_profile_consistency_loss(image, template, vessel_flow, vessel_radius):
    """ Profile consistency loss Lm
    image: I
    template: T
    vessel_flow: u
    vessel_radius: r
    """
    loss = -np.sum(get_vesselness(image, [template, vessel_flow, vessel_radius]) * dp)
    return loss


#
def get_path_continuity_loss():
    """Path continuity loss Lf

    :return:
    """
    return 0


# Bifurcation loss Lb
def get_bifurcation_loss(image, template, bifurcation_flow_fields, vessel_radius):
    loss = 0
    for vessel_flow in bifurcation_flow_fields:
        loss += get_profile_consistency_loss(
            image, template, vessel_flow, vessel_radius
        )
    return loss


#
def get_regularizers(image, template, vessel_flow, vessel_radius):
    """
    Regularizers for vesselness
    :param image:
    :param template:
    :param vessel_flow:
    :param vessel_radius:
    :return:
    """
    return 0


def similarity_function(a, b):
    # Normalized cross correlation
    a = (a - np.mean(a)) / (np.std(a) * len(a))
    b = (b - np.mean(b)) / (np.std(b))
    return np.correlate(a, b, "full")


def get_vesselness(image, templates, vessel_flow, vessel_radius):
    vesselness = sys.maxsize
    p = get_profile(image, vessel_flow, vessel_radius)
    for template in templates:
        vesselness = min(similarity_function(p, template), vesselness)
    return vesselness


def overall_cost(image, template, vessel_flows, vessel_radius):
    cost = 0
    for vessel_flow in vessel_flows:
        cost += get_profile_consistency_loss(
            image, template, vessel_flow, vessel_radius
        )
        cost += HParams.lambda1 * get_path_continuity_loss()
        cost += HParams.lambda3 * get_regularizers(
            image, template, vessel_flow, vessel_radius
        )
    return cost + HParams.lambda2 * get_bifurcation_loss(
        image, template, vessel_flows, vessel_radius
    )
