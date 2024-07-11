import torch 
import random
import numpy as np
import math 
from tqdm import tqdm, trange
from torch_geometric.nn import GCNConv
from layers import AttentionModule, TenorNetworkModule
from utils import  calculate_loss, calculate_normalized_gosim, train_test_split
from dataset import TopologicalDataset, BiologicalDataset

random.seed(42)
torch.manual_seed(42)

class GBNSS(torch.nn.Module):

    def __init__(self, args, features_size):
        """
        :param args: Arguments object.
        :param feature_size: number of node features(input features)
        """
        super(GBNSS, self).__init__()
        self.args = args
        self.feature_size = features_size
        self.setup_layers()
    
    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram == True:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        #Creating the layers
        
        self.calculate_bottleneck_features()
        self.convolution_1 = GCNConv(self.feature_size, self.args.filters_1)
        self.convolution_2 = GCNConv(self.args.filters_1, self.args.filters_2)
        self.convolution_3 = GCNConv(self.args.filters_2, self.args.filters_3)
        self.attention = AttentionModule(self.args)
        self.tensor_network = TenorNetworkModule(self.args)
        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.args.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.args.bottle_neck_neurons, 1)                                             

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histsogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist/torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Absstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_2(features, edge_index)                                       
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_index)
        return features
    
    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictiyonary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index_1"]
        edge_index_2 = data["edge_index_2"]
        
        features_1 = data["features_1"]
        features_2 = data["features_2"]
        
        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)
        
        if self.args.histogram == True:
            hist = self.calculate_histogram(abstract_features_1,
                                            torch.t(abstract_features_2))
            
        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)

        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        if self.args.histogram == True:
            scores = torch.cat((scores, hist), dim=1).view(1, -1)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores))
        
        return score


class GBNSS_Topological_Trainer(object):
    """
    GBNSS model trainer
    """
    
    def __init__(self, args):
        """
        :param args: Arguments object.
        """

        self.args = args
        self.initial_data()
        self.setup_model()

    def setup_model(self):
        self.model = GBNSS(self.args, self.feature_size)

    def initial_data(self):
        print("Loading datasets...")
        self.dataset = TopologicalDataset(root="./../dataset/data.csv").process()
        self.train_dataset, self.test_dataset = train_test_split(self.dataset, 0.7)
        self.feature_size = self.train_dataset[0]["features_1"].shape[1]

    def create_train_batches(self):
        """
        Creating batches from the training graph list.
        :return batched: List of lists with batches.
        """
        random.shuffle(self.train_dataset)
        batches = []
        for graph in range(0, len(self.train_dataset), self.args.batch_size):
            batches.append(self.train_dataset[graph:graph+self.args.batch_size])
        return batches
    
    def create_test_batches(self):
        """
        Creating batches from the testing graph list.
        :return batched: List of lists with batches.
        """
        random.shuffle(self.test_dataset)
        batches = []
        for graph in range(0, len(self.test_dataset), self.args.batch_size):
            batches.append(self.test_dataset[graph:graph+self.args.batch_size])
        return batches
       
    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair loactions.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        losses = 0
        for graph_pair in batch:
            target = graph_pair[self.args.label]
            pred = self.model(graph_pair)
            target = target.view_as(pred)
            losses = losses + torch.nn.functional.mse_loss(target, pred)
        losses.backward(retain_graph=True)
        self.optimizer.step()
        loss = losses.item()
        return loss

    def fit(self):
        """
        Fitting a model
        """
        print("\nModel training.\n")

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr = self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        
        epochs = trange(self.args.epochs, leave=True, desc="Epoch")

        self.model.train()
        for epoch in epochs:
            batches = self.create_train_batches()
            self.loss_sum = 0
            main_index = 0
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                loss_score = self.process_batch(batch)
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score * len(batch)
                loss = self.loss_sum/main_index
                epochs.set_description("Epoch (Train Loss=%g)" % round(loss, 5))
                    
    def score(self):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation.\n")
        self.model.eval()
        self.scores = []
        self.ground_truth = []
        for graph_pair in tqdm(self.test_dataset):
            target = graph_pair[self.args.label]
            ground_truth = -math.log(target)*(0.5*(len(graph_pair["features_1"]) + len(graph_pair["features_2"])))
            self.ground_truth.append(ground_truth)
            pred = self.model(graph_pair)
            self.scores.append(calculate_loss(pred, target))
        self.print_evaluation()

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        ground_truth = np.asarray(self.ground_truth, dtype=np.float32)
        norm_ged_mean = np.mean(ground_truth)
        base_error = np.mean([(n-norm_ged_mean)**2 for n in ground_truth])
        model_error = np.mean(self.scores)
        print("\nModel test error: " + str(round(model_error, 5)) + ".")

    def save(self):
        torch.save(self.model.state_dict(), self.args.save_path)
    

class GBNSS_Biological_Trainer(object):
    """
    GBNSS model trainer
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        self.args = args
        self.initial_data()
        self.setup_model()

    def setup_model(self):
        self.model = GBNSS(self.args, self.feature_size)

    def initial_data(self):
        print("Loading datasets...")
        self.dataset = BiologicalDataset(root="./../dataset/data.csv").process()
        self.train_dataset, self.test_dataset = train_test_split(self.dataset, 0.7)
        self.feature_size = self.train_dataset[0]["features_1"].shape[1]

    def create_train_batches(self):
        """
        Creating batches from the training graph list.
        :return batched: List of lists with batches.
        """
        random.shuffle(self.train_dataset)
        batches = []
        for graph in range(0, len(self.train_dataset), self.args.batch_size):
            batches.append(self.train_dataset[graph:graph+self.args.batch_size])
        return batches
    
    def create_test_batches(self):
        """
        Creating batches from the testing graph list.
        :return batched: List of lists with batches.
        """
        random.shuffle(self.test_dataset)
        batches = []
        for graph in range(0, len(self.test_dataset), self.args.batch_size):
            batches.append(self.test_dataset[graph:graph+self.args.batch_size])
        return batches
      
    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair loactions.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        losses = 0
        for graph_pair in batch:
            target = graph_pair[self.args.label]
            pred = self.model(graph_pair)
            target = target.view_as(pred)
            losses = losses + torch.nn.functional.mse_loss(target, pred)
        losses.backward(retain_graph=True)
        self.optimizer.step()
        loss = losses.item()
        return loss

    def fit(self):
        """
        Fitting a model
        """
        print("\nModel training.\n")

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr = self.args.learning_rate,
                                          weight_decay=self.args.weight_decay)
        
        epochs = trange(self.args.epochs, leave=True, desc="Epoch")

        self.model.train()
        for epoch in epochs:
            batches = self.create_train_batches()
            self.loss_sum = 0
            main_index = 0
            for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
                loss_score = self.process_batch(batch)
                main_index = main_index + len(batch)
                self.loss_sum = self.loss_sum + loss_score * len(batch)
                loss = self.loss_sum/main_index
                epochs.set_description("Epoch (Train Loss=%g)" % round(loss, 5))
                   
    def score(self):
        """
        Scoring on the test set.
        """
        print("\n\nModel evaluation.\n")
        self.model.eval()
        self.scores = []
        self.ground_truth = []
        for graph_pair in tqdm(self.test_dataset):
            target = graph_pair[self.args.label]
            ground_truth = -math.log(target)*(0.5*(len(graph_pair["features_1"]) + len(graph_pair["features_2"])))
            self.ground_truth.append(ground_truth)
            pred = self.model(graph_pair)
            self.scores.append(calculate_loss(pred, target))
        self.print_evaluation()

    def print_evaluation(self):
        """
        Printing the error rates.
        """
        ground_truth = np.asarray(self.ground_truth, dtype=np.float32)
        norm_ged_mean = np.mean(ground_truth)
        base_error = np.mean([(n-norm_ged_mean)**2 for n in ground_truth])
        model_error = np.mean(self.scores)
        print("\nModel test error: " + str(round(model_error, 5)) + ".")

    def save(self):
        torch.save(self.model.state_dict(), self.args.save_path)        

