from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEmbedding(nn.Module):
    """
    A feature embedding that applies a random Fourier transform to inputs.
    Often used to help the network learn high-frequency functions.
    """

    def __init__(self, input_dim, num_fourier_features, scale=1.0, dtype=torch.float32):
        super().__init__()
        self.input_dim = input_dim
        self.num_fourier_features = num_fourier_features
        self.scale = scale
        self.dtype = dtype

        # Create a random projection matrix: (input_dim x num_fourier_features)
        # This projects 'inputs' into a higher-frequency space.
        # Typically this is not learned, but you can make it a parameter if desired.
        B = torch.randn(input_dim, num_fourier_features, dtype=dtype) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape (batch_size, input_dim)
        Returns a concatenation of sin(x @ B) and cos(x @ B), shape (batch_size, 2 * num_fourier_features).
        """
        # (batch_size, num_fourier_features)
        x_proj = x.matmul(self.B)
        # Apply sin/cos
        x_sin = torch.sin(x_proj)
        x_cos = torch.cos(x_proj)
        # Concatenate along the last dimension
        return torch.cat([x_sin, x_cos], dim=-1)


class NNWithFourier(nn.Module):
    """
    A neural network that first embeds inputs using random Fourier features,
    then processes them with an MLP. Similar structure to your provided NN,
    but with a FourierEmbedding in front.
    """

    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_fourier_features: int = 16,
        fourier_scale: float = 1.0,
        dtype: torch.dtype = torch.float32,
        activation: nn.Module = nn.Tanh()
    ):
        super().__init__()
        self.dtype = dtype

        # 1) Create the Fourier embedding layer
        #    Outputs dimension = 2 * num_fourier_features
        self.fourier = FourierEmbedding(
            input_dim=input_dim,
            num_fourier_features=num_fourier_features,
            scale=fourier_scale,
            dtype=dtype
        )

        # 2) Build the MLP layers
        #    First layer receives the Fourier-embedded features as input
        layers = []
        fourier_out_dim = 2 * num_fourier_features  # sin + cos
        layers.append(nn.Linear(fourier_out_dim, hidden_dim, dtype=dtype))
        layers.append(activation)

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, dtype=dtype))
            layers.append(activation)

        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim, dtype=dtype)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # 1) Embed inputs with Fourier features
        x_emb = self.fourier(inputs)

        # 2) Pass through hidden layers
        h = self.hidden_layers(x_emb)

        # 3) Final linear layer
        out = self.output_layer(h)
        return out


class NN(nn.Module):
    """
        Vanilla NN.
    """

    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, dtype=torch.float32, activation=nn.SiLU()):
        super(NN, self).__init__()
        layers = []
        # First layer: Linear followed by activation
        layers.append(nn.Linear(input_dim, hidden_dim, dtype=dtype))
        layers.append(activation)

        # Additional hidden layers: each Linear
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, dtype=dtype))
            layers.append(activation)

        # Combine all the hidden layers into a single sequential module
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim, dtype=dtype)

    def forward(self, inputs):
        x = self.hidden_layers(inputs)
        x = self.output_layer(x)
        return x


class NNWithAnsatz(nn.Module):
    """
        NN with a differentiable approximation of the payoff function.
    """

    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, dtype=torch.float32, activation=nn.Sigmoid()):
        super(NNWithAnsatz, self).__init__()
        layers = []
        self.input_dim = input_dim
        # First layer: Linear followed by activation
        layers.append(nn.Linear(input_dim, hidden_dim, dtype=dtype))
        layers.append(activation)
        self.alpha = nn.Parameter(torch.tensor([10.0]))
        self.beta = nn.Parameter(torch.tensor([10.0]))

        # Additional hidden layers: each Linear +  activation
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, dtype=dtype))
            layers.append(activation)

        # Combine all the hidden layers into a single sequential module
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim, dtype=dtype)

    def payoff(self, x):
        # x = torch.max(torch.exp(x), dim=1).values
        x = self.smooth_max_exp(x, alpha=self.alpha)
        ones = torch.ones_like(x)
        # zeros = torch.zeros_like(x)
        # payoff_values = torch.maximum(x - ones, zeros)
        payoff_values = F.gelu(x - ones)
        return payoff_values.reshape([-1, 1])

    def smooth_max_exp(self, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Differentiable approximation of max(exp(x_j)) across the last dimension of x
          smooth_max_exp(x, alpha) = ( sum_j exp(alpha * (x_j - x_max)) )^(1/alpha) * exp(x_max)
        where x_max is the max across each row to improve numerical stability.
        """
        x_max = x.max(dim=1, keepdim=True).values
        # log-sum-exp for alpha*(x - x_max)
        lse = torch.logsumexp(alpha * (x - x_max), dim=1, keepdim=True)
        # exponentiate
        return torch.exp((lse + alpha * x_max) / alpha).squeeze(1)

    def forward(self, inputs):
        x = self.hidden_layers(inputs)
        x = self.output_layer(x)
        p = self.payoff(inputs[:, :self.input_dim-1])
        return p + x


class SPINN(nn.Module):
    """
        SPINN is a model that learns a tensor decomposition of the solution instead of directly trying to learn the solution.
        Works well but requires more computing power.

        It has a Vanilla NN per dimension, and the output is computed as a dot product of the output of all NN.

        https://arxiv.org/abs/2306.15969
    """

    def __init__(self, n_layers, input_dim, hidden_dim, output_dim, dtype=torch.float32):
        super(SPINN, self).__init__()
        self.n_dim = input_dim
        # Create one NN per input dimension. Each NN accepts a single feature.
        self.sub_nns = nn.ModuleList([
            NN(n_layers, input_dim=1, hidden_dim=hidden_dim,
               output_dim=output_dim, dtype=dtype)
            for _ in range(input_dim)
        ])

    def forward(self, inputs):
        outputs = []
        for i, sub_nn in enumerate(self.sub_nns):
            # Extract the i-th dimension and ensure it has shape (batch_size, 1)
            x_i = inputs[:, i].unsqueeze(1)
            # each output has shape (batch_size, hidden_dim)
            outputs.append(sub_nn(x_i))

        # Combine outputs by taking an elementwise product
        prod = outputs[0]
        for out in outputs[1:]:
            prod = prod * out  # elementwise multiplication

        # Reduce the product over the hidden_dim to produce a scalar for each sample
        scalar_output = prod.sum(dim=1, keepdim=True)
        return scalar_output


def weights_init(m):
    """
    Custom weights initialization for nn.Module.
    """
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.0)


def build_nn(nn_shape: str, input_dim: int, dtype=torch.float32, activation=nn.Tanh()):
    """
    Build a neural network based on the provided shape string.
    """
    try:
        h, l = nn_shape.split('x')
        hidden_dim = int(h)
        n_layers = int(l)
    except Exception:
        raise ValueError(f"Unknown nn_shape: {nn_shape}")

    model = NN(n_layers=n_layers,
               input_dim=input_dim + 1,
               hidden_dim=hidden_dim,
               output_dim=1,
               dtype=dtype, activation=activation)
    model.apply(weights_init)
    return model


class FirstOrderNN(nn.Module):

    def __init__(self,
                 n_layers: int,
                 input_dim: int,
                 hidden_dim: int,
                 activation: Optional[nn.Module] = nn.Tanh(),
                 dtype=torch.float32):

        super().__init__()
        layers = []
        # first hidden layer
        layers.append(nn.Linear(input_dim, hidden_dim, dtype=dtype))
        layers.append(activation)
        # remaining hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, dtype=dtype))
            layers.append(activation)

        self.hidden_layers = nn.Sequential(*layers)
        # final output: 1 (u) + d (predicted ∂u/∂x_i)
        self.output_layer = nn.Linear(hidden_dim, 1 + input_dim, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        h = self.hidden_layers(x)
        out = self.output_layer(h)
        return out
