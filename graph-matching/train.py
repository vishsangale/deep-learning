import torch

from model import MatchingGCN

from dataset import SyntheticDataLoader, SyntheticGraphDataset

def euclidean_distance(x, y):
    """This is the squared Euclidean distance."""
    return torch.sum((x - y) ** 2, dim=-1)


def approximate_hamming_similarity(x, y):
    """Approximate Hamming similarity."""
    return torch.mean(torch.tanh(x) * torch.tanh(y), dim=1)

def pairwise_loss(x, y, labels, loss_type='margin', margin=1.0):
    """Compute pairwise loss.
    Args:
      x: [N, D] float tensor, representations for N examples.
      y: [N, D] float tensor, representations for another N examples.
      labels: [N] int tensor, with values in -1 or +1.  labels[i] = +1 if x[i]
        and y[i] are similar, and -1 otherwise.
      loss_type: margin or hamming.
      margin: float scalar, margin for the margin loss.
    Returns:
      loss: [N] float tensor.  Loss for each pair of representations.
    """

    labels = labels.float()

    if loss_type == 'margin':
        return torch.relu(margin - labels * (1 - euclidean_distance(x, y)))
    elif loss_type == 'hamming':
        return 0.25 * (labels - approximate_hamming_similarity(x, y)) ** 2
    else:
        raise ValueError('Unknown loss_type %s' % loss_type)

# m = MNIST()
loader = SyntheticDataLoader(
    SyntheticGraphDataset(),


)
model = MatchingGCN(in_channels=16)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for idx, data in enumerate(loader):
    print(data)
    prediction = model(data)     # input x and predict based on x

    loss = pairwise_loss(prediction, idx//2==0, idx//2==0)     # must be (1. nn output, 2. target)

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()
