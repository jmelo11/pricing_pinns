import torch.nn as nn
import torch


class BranchNet(nn.Module):
    def __init__(self, param_dim, hidden_layers=[20, 20], latent_dim=50, dtype=torch.float64):
        super(BranchNet, self).__init__()
        layers = []
        in_dim = param_dim
        # Build hidden layers
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h, dtype=dtype))
            layers.append(nn.Tanh())  # activation for nonlinearity
            in_dim = h
        # Final layer to latent_dim (no activation here)
        layers.append(nn.Linear(in_dim, latent_dim, dtype=dtype))
        self.model = nn.Sequential(*layers)

    def forward(self, params):
        return self.model(params)


class TrunkNet(nn.Module):
    def __init__(self, coord_dim, hidden_layers=[20, 20], latent_dim=50, dtype=torch.float64):
        super(TrunkNet, self).__init__()
        layers = []
        in_dim = coord_dim
        for h in hidden_layers:
            layers.append(nn.Linear(in_dim, h, dtype=dtype))
            layers.append(nn.Tanh())
            in_dim = h
        layers.append(nn.Linear(in_dim, latent_dim, dtype=dtype))
        self.model = nn.Sequential(*layers)

    def forward(self, coords):
        return self.model(coords)


class DeepONetPINN(nn.Module):
    def __init__(self, param_dim, hidden_layers, coord_dim, latent_dim=50, dtype=torch.float64):
        super(DeepONetPINN, self).__init__()
        self.branch_net = BranchNet(
            param_dim, hidden_layers, latent_dim=latent_dim, dtype=dtype)
        self.trunk_net = TrunkNet(
            coord_dim, hidden_layers, latent_dim=latent_dim, dtype=dtype)

    def forward(self, params, coords):
        """
        Forward pass for PINN.
        params: tensor of shape (N, param_dim)
        coords: tensor of shape (N, coord_dim) corresponding to (t, S, r, v)
        """
        # Compute branch and trunk features
        branch_features = self.branch_net(params)    # shape (N, latent_dim)
        trunk_features = self.trunk_net(coords)     # shape (N, latent_dim)

        output = (branch_features * trunk_features).sum(dim=1,
                                                        keepdim=True).squeeze(1)

        return output


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_layers):
        super().__init__()
        layers = []
        prev_dim = in_dim

        for h in hidden_layers:
            layers += [nn.Linear(prev_dim, h, dtype=torch.float64), nn.Softplus()]
            prev_dim = h

        layers.append(nn.Linear(prev_dim, out_dim, dtype=torch.float64))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze()
