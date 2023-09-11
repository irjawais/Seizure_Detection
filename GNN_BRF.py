import numpy as np
from imblearn.over_sampling import SMOTE
from torch_geometric.nn import GCNConv
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split
from scipy.sparse import coo_matrix
import networkx as nx
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import networkx as nx
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import seaborn as sns
from scipy.spatial.distance import euclidean
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score, roc_auc_score, auc
from torchviz import make_dot
import netron
import scikitplot as skplt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from pyswarms.discrete.binary import BinaryPSO
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
import scipy.sparse as sp
from knnor import data_augment
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def create_adj_matrix(graph):
    num_nodes = len(graph)
    adj_matrix = [[0 for i in range(num_nodes)] for j in range(num_nodes)]

    for node in graph:
        for neighbor in graph[node]:
            adj_matrix[node][neighbor] = 1
    return adj_matrix

def fit(self, x, y):
    x = x.numpy()
    y = y.numpy()
    self.clf.fit(x, y)

# Define the GCN model
class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)
        x = torch.relu(x)
        x = self.conv4(x, edge_index)
        x = torch.relu(x)
        x = self.conv5(x, edge_index)
        x = torch.relu(x)
        return x
        
def create_graph(X, y):

    # Create an empty graph
    G = nx.Graph()
    # Add each row as a node in the graph
    for i in range(X.shape[0]):
        G.add_node(i)
    # Loop through all nodes to create edges
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            if i == j:
                continue
            dist = euclidean(X[i], X[j])

            if dist < 0.2 : #
                G.add_edge(i, j)
    return  G

each_channel_features = np.load('content/_each_channel_features.npy', allow_pickle=True)
each_channel_features = np.nan_to_num(each_channel_features, nan=0, posinf=0, neginf=0)
print(each_channel_features.shape)
selected_1_rows = each_channel_features[each_channel_features[:, -1] == 1]
selected_0_rows = each_channel_features[each_channel_features[:, -1] == 0]
#selected_0_rows = selected_0_rows[:100]
merged_rows = np.concatenate((selected_1_rows, selected_0_rows), axis=0)

#each_channel_features = each_channel_features[10000:20000]

''' print("befiore ",merged_rows.shape)
new_data = np.loadtxt('new_data.csv', delimiter=',')
zeros_col = np.zeros((new_data.shape[0], 1))
new_data = np.hstack((new_data, zeros_col))
merged_rows = np.vstack((merged_rows, new_data))
print("after ",merged_rows.shape) '''

X = merged_rows[:, :-1]
y = merged_rows[:, -1]


''' smote = SMOTE(sampling_strategy='minority')
X, y = smote.fit_resample(X, y) '''

''' knnor = data_augment.KNNOR()
X, y, X_aug_min, y_aug_min = knnor.fit_resample(X,y) '''


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)



G = create_graph(X, y)
labels = y

adj = nx.adjacency_matrix(G)
adj = adj.todense()
adj = np.array(adj)


labels = labels.astype(int)
labels = torch.from_numpy(labels).long()
print(len(adj),len(labels))
# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(adj, labels, test_size=0.01, random_state=42)
print("===>",G.number_of_nodes())
# Create the GCN + LSTM model
model = GCN(in_channels=G.number_of_nodes(), hidden_channels=256, out_channels=16)


num_nodes = X_train.shape[0]
valid_indices = np.where((X_train.nonzero()[0] < num_nodes) & (X_train.nonzero()[1] < num_nodes))
edge_index = torch.from_numpy(np.concatenate((X_train.nonzero()[0][valid_indices][np.newaxis, :],
                                                X_train.nonzero()[1][valid_indices][np.newaxis, :]), axis=0))
X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)

X_train = torch.tensor(X_train.float()).clone().detach()
X_test = torch.tensor(X_test.float()).clone().detach()
y_train = torch.tensor(y_train.float()).clone().detach()
y_test = torch.tensor(y_test.float()).clone().detach()

X_train = X_train.float().clone().detach().requires_grad_(True)
X_test = X_test.float().clone().detach().requires_grad_(True)
y_train = y_train.float().clone().detach().requires_grad_(True)
y_test = y_test.float().clone().detach().requires_grad_(True)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=5e-4)
''' device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights = [0.02,0.98]  # weights for 16 classes, adjust according to your dataset
weight_tensor = torch.FloatTensor(weights).to(device)  # convert to tensor and move to device '''

criterion = nn.CrossEntropyLoss()
# Train the GCN model
for epoch in range(2000):
    optimizer.zero_grad()
    out = model(X_train.float(), edge_index)
    loss = criterion(out, y_train.long())
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')

out = out.detach().numpy()
y_train = y_train.detach().numpy()

# Extract the node embeddings
''' with torch.no_grad():
    embeddings = model.get_embedding(X_train.float(), edge_index).numpy() '''


# Use the GCN model to transform the data
''' with torch.no_grad():
    X_train_enhanced = model(X_train.float(), edge_index)
    X_test_enhanced = model(X_test.float(), edge_index) '''



# Create a balanced random forest classifier and train it on the enhanced data
clf = BalancedRandomForestClassifier()
clf.fit(out, y_train)

# Evaluate the classifier
acc = clf.score(out, y_train)
print(f"Accuracy: {acc}")
from sklearn.metrics import classification_report

# Predict labels for the test data
y_pred = clf.predict(out)

# Compute classification metrics
report = classification_report(y_pred, y_train, output_dict=True)
print(report)

# create the confusion matrix
cm = confusion_matrix(y_pred, y_train)
# plot the confusion matrix using seaborn
sns.heatmap(cm, annot=True, cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Visualize the node embeddings using t-SNE
pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(out)
plt.scatter(embeddings_pca[:,0], embeddings_pca[:,1], c=y_train)
plt.show()
pca = PCA(n_components=2)
embeddings_pca = pca.fit_transform(out)
plt.scatter(embeddings_pca[:,0], embeddings_pca[:,1], c=y_train)
plt.show()

tsne = TSNE(n_components=2)
embeddings_tsne = tsne.fit_transform(out)
plt.scatter(embeddings_tsne[:,0], embeddings_tsne[:,1], c=y_train)
plt.show()

''' num_nodes = X_train.shape[0]
valid_indices = np.where((X_train.nonzero()[0] < num_nodes) & (X_train.nonzero()[1] < num_nodes))
edge_index = torch.from_numpy(np.concatenate((X_train.nonzero()[0][valid_indices][np.newaxis, :],
                                                X_train.nonzero()[1][valid_indices][np.newaxis, :]), axis=0))
                                                
adj = torch.from_numpy(adj)

X_train = torch.from_numpy(X_train)
X_test = torch.from_numpy(X_test)

X_train = torch.tensor(X_train.float()).clone().detach()
X_test = torch.tensor(X_test.float()).clone().detach()
y_train = torch.tensor(y_train.float()).clone().detach()
y_test = torch.tensor(y_test.float()).clone().detach()

X_train = X_train.float().clone().detach().requires_grad_(True)
X_test = X_test.float().clone().detach().requires_grad_(True)
y_train = y_train.float().clone().detach().requires_grad_(True)
y_test = y_test.float().clone().detach().requires_grad_(True)



print(X_train.shape,G.number_of_nodes(),edge_index.shape)
accuracies = []
losses = []
recalls = []
precisions = []
precisions
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001,weight_decay=5e-4)

criterion = nn.CrossEntropyLoss()

y_train_pred = []
for epoch in range(3000):
    optimizer.zero_grad()
    out = model(X_train.float(), edge_index, y_train.float())
    loss = criterion(out, y_train.float())
    loss.backward()
    optimizer.step()
    _, predicted = torch.max(out.data, 0)
    correct = (predicted == y_train).sum().item()
    accuracy = correct / len(y_train)
    accuracies.append(accuracy)
    print(accuracy) 
    print("predicted" , y_train.shape, predicted.shape)
    tp, fp, tn, fn = confusion_matrix(y_train, predicted, labels=[1, 0]).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    recalls.append(sensitivity)
    precision = tp / (tp + fp)
    precisions.append(precision)
    f1 = f1_score(y_train, predicted)
    losses.append(loss.item())
    k = cohen_kappa_score(y_train, predicted)
    auc_value = roc_auc_score(y_train, predicted)
    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, Precision: {precision:.4f}, F1-measure: {f1:.4f}, Kappa: {k:.4f}, AUC: {auc_value:.4f}')
 '''

''' # define the Balanced Random Forest classifier
clf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)

# train the classifier on the training set
clf.fit(X_train, y_train)

# predict on the testing set
y_pred = clf.predict(X_test)

# evaluate the classifier's performance
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))



from sklearn.model_selection import StratifiedKFold
from imblearn.ensemble import BalancedRandomForestClassifier

# create an instance of StratifiedKFold
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# create lists to store the results
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# loop over the splits
for train_index, test_index in skf.split(X, y):
    # split the data into train and test sets
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # create an instance of the BalancedRandomForestClassifier
    clf = BalancedRandomForestClassifier(n_estimators=100, random_state=42)
    
    # fit the model on the training set
    clf.fit(X_train, y_train)
    
    # make predictions on the test set
    y_pred = clf.predict(X_test)
    
    # calculate the evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # append the results to the lists
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)

    skplt.metrics.plot_confusion_matrix(y_test,y_pred, normalize=True, cmap=plt.cm.Blues)
    plt.tight_layout()
    plt.show()
    
# calculate the average scores across the splits
avg_accuracy = np.mean(accuracy_scores)
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1 = np.mean(f1_scores)

# print the results
print("Average accuracy:", avg_accuracy)
print("Average precision:", avg_precision)
print("Average recall:", avg_recall)
print("Average F1-score:", avg_f1) '''
