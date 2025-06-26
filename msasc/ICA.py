import torch
import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler


def I_acoustic(acoustic):
    # ICA解耦
    acoustic = acoustic.view(acoustic.size(0), -1)
    n_components = 74

    # 使用稀疏因子分析（Sparse Factor Analysis）
    scaler = StandardScaler()
    acoustic_scaled = scaler.fit_transform(acoustic.numpy())
    fa = FactorAnalysis(n_components=n_components)
    ica_acoustic_ids = fa.fit_transform(acoustic_scaled)

    ica_acoustic = torch.tensor(ica_acoustic_ids)
    ica_acoustic = ica_acoustic.unsqueeze(dim=2)

    return ica_acoustic


def I_visual(visual):
    visual = visual.view(visual.size(0), -1)
    n_components = 74

    # 使用稀疏因子分析（Sparse Factor Analysis）
    scaler = StandardScaler()
    visual_scaled = scaler.fit_transform(visual.numpy())
    fa = FactorAnalysis(n_components=n_components)
    ica_visual_ids = fa.fit_transform(visual_scaled)

    ica_visual = torch.tensor(ica_visual_ids)
    ica_visual = ica_visual.unsqueeze(dim=2)

    return ica_visual
