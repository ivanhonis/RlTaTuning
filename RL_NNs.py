from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch as th
import torch.nn as nn


class CustomLinear(BaseFeaturesExtractor):
    """
    A simple custom linear feature extractor for Stable-Baselines3 (SB3) with two layers,
    each having 16 units, using ReLU activation and Dropout for regularization.
    """

    def __init__(self, observation_space, features_dim: int = 64):
        super(CustomLinear, self).__init__(observation_space, features_dim)

        # Flatten the input features if needed
        # n_input_features = observation_space.shape[0] * observation_space.shape[1]
        n_input_features = observation_space.shape[0]

        # Define the fully connected layers with Dropout
        self.linear1 = nn.Linear(n_input_features, 16)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)  # Dropout layer (50% chance of dropping units)

        self.linear2 = nn.Linear(16, 16)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)  # Dropout layer (50% chance of dropping units)

        # Final fully connected layer to map the output to features_dim
        self.fc = nn.Linear(16, features_dim)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Flatten the input to (batch_size, n_input_features)
        x = observations.flatten(start_dim=1)

        # First fully connected layer with ReLU and Dropout
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        # Second fully connected layer with ReLU and Dropout
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        # Final output layer
        x = self.fc(x)
        return x


class CustomCNN3(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 64):
        super(CustomCNN3, self).__init__(observation_space, features_dim)

        # Extract the number of input channels (features)
        n_input_channels = observation_space.shape[0]  # Observation space shape (channels, time_steps)
        # time_steps = observation_space.shape[1]        # Length of the time sequence (600 if 10 minutes of 1 Hz data)

        # Define each layer individually
        self.conv1 = nn.Conv1d(in_channels=n_input_channels, out_channels=32, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm1d(32)  # Batch normalization after first conv layer
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(32)  # Batch normalization after second conv layer
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.flatten = nn.Flatten()  # Flatten layer to convert 3D tensor to 2D

        # Compute the output size after passing through CNN layers
        with th.no_grad():
            sample_input = th.zeros((1,) + observation_space.shape)
            n_flatten = self.forward_cnn(sample_input).shape[1]

        self.fc1 = nn.Linear(n_flatten, 32)  # First fully connected layer with 128 units
        self.fc2 = nn.Linear(32, features_dim)  # Second fully connected layer, mapping to features_dim
        self.relu = nn.ReLU()

    def forward_cnn(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)  # Flatten the output for fully connected layer
        return x

    def forward(self, observations: th.Tensor) -> th.Tensor:
        cnn_output = self.forward_cnn(observations)

        fc_output = self.fc1(cnn_output)
        fc_output = self.relu(fc_output)
        fc_output = self.fc2(fc_output)
        return self.relu(fc_output)


class CustomCNN2(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super(CustomCNN2, self).__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[1]
        # print(observation_space.shape)

        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_input_channels, out_channels=8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(8),  # Optional batch normalization
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=8, out_channels=8, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.BatchNorm1d(8),  # Optional batch normalization
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )

        with th.no_grad():
            sample_input = th.zeros((1,) + observation_space.shape)
            n_flatten = self.cnn(sample_input).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
            # nn.Dropout(p=0.15)  # Optional dropout
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        cnn_output = self.cnn(observations)
        return self.fc(cnn_output)


class CustomLSTM(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 64, lstm_hidden_size: int = 128):
        # Call the parent constructor
        super(CustomLSTM, self).__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[1]  # The number of input channels (e.g., features)
        seq_length = observation_space.shape[0]  # Length of the sequence (e.g., timesteps)

        # LSTM Layer: input is expected to be (batch_size, seq_length, input_size)
        self.lstm = nn.LSTM(
            input_size=n_input_channels,  # The number of input features per time step
            hidden_size=lstm_hidden_size,  # Number of LSTM units in the hidden layer
            batch_first=True  # The input shape is (batch_size, seq_length, input_size)
        )

        # Flatten the LSTM output and connect it to a fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size * seq_length, features_dim),
            # Linear layer to reduce the LSTM output to `features_dim`
            nn.ReLU(),
            # nn.Dropout(p=0.15)  # Optional dropout for regularization
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Pass input through LSTM
        # Input shape: (batch_size, seq_length, n_input_channels)
        lstm_output, (_hidden_state, _cell_state) = self.lstm(observations)

        # Flatten the LSTM output for fully connected layers
        lstm_output = lstm_output.reshape(lstm_output.size(0), -1)

        # Pass through fully connected layers
        return self.fc(lstm_output)


class CustomCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space, features_dim=64):
        # Initialize the BaseFeaturesExtractor with the observation space
        super(CustomCNN, self).__init__(observation_space, features_dim)

        # Extract the number of input channels from the observation space
        n_input_channels = observation_space.shape[0]

        # Define a simple CNN with 1D convolution
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_input_channels, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Flatten()
        )

        # Compute the size of the output from the CNN
        with th.no_grad():
            sample_input = th.zeros((1,) + observation_space.shape)
            n_flatten = self.cnn(sample_input).shape[1]

        # Define a fully connected layer to map the CNN output to the features_dim
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # Pass the observations through the CNN
        cnn_output = self.cnn(observations)
        # Pass the output of the CNN through the fully connected layer
        return self.fc(cnn_output)
