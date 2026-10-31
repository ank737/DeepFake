import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------
# 1. Separable Convolution
# --------------------------------

class SeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):

        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels
        )

        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1
        )

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):

        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)

        return F.relu(x)


# --------------------------------
# 2. Graph Convolution
# --------------------------------

class GraphConv(nn.Module):

    def __init__(self, in_features, out_features):

        super().__init__()

        self.linear = nn.Linear(
            in_features,
            out_features
        )

    def forward(self, x, adjacency):

        x = torch.bmm(adjacency, x)
        x = self.linear(x)

        return F.relu(x)


# --------------------------------
# 3. Hybrid CNN-GCN
# --------------------------------

class HybridGCN(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        # CNN branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        self.blocks = nn.Sequential(
            SeparableConv(32, 32),
            SeparableConv(32, 64, stride=2),
            SeparableConv(64, 64),
            SeparableConv(64, 128, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        # Graph branch
        self.gcn1 = GraphConv(128, 64)
        self.gcn2 = GraphConv(64, 32)

        self.graph_conv1d = nn.Conv1d(
            32, 32, kernel_size=3, padding=1
        )

        # Classification
        self.classifier = nn.Sequential(
            nn.Linear(128 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        # CNN branch
        x = self.conv1(x)
        x = F.relu(self.conv2(x))
        x = self.blocks(x)

        x = self.conv3(x)
        x = self.conv4(x)

        cnn_features = self.pool(x).flatten(1)

        # Convert spatial feature map into graph nodes
        nodes = x.flatten(2).transpose(1, 2)

        batch_size, num_nodes, _ = nodes.shape

        # Approximate four-neighbour graph using
        # self-loops and adjacent nodes in flattened order
        adjacency = torch.eye(
            num_nodes, device=x.device
        )

        adjacency = adjacency.clone()

        for i in range(num_nodes - 1):
            adjacency[i, i + 1] = 1
            adjacency[i + 1, i] = 1

        adjacency = adjacency / adjacency.sum(
            dim=1, keepdim=True
        )

        adjacency = adjacency.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # GCN branch
        graph_features = self.gcn1(nodes, adjacency)
        graph_features = self.gcn2(
            graph_features, adjacency
        )

        graph_features = graph_features.transpose(1, 2)

        graph_features = F.relu(
            self.graph_conv1d(graph_features)
        )

        # Approximate sort pooling using max pooling
        graph_features = torch.max(
            graph_features, dim=2
        )[0]

        # Feature concatenation
        combined = torch.cat(
            [cnn_features, graph_features], dim=1
        )

        return self.classifier(combined)