
from __future__ import absolute_import
import json
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os, sys, csv
from dgl import backend as F
from dgl.convert import from_networkx
from dgl.data.dgl_dataset import DGLDataset
from dgl.data.utils import download, save_graphs, load_graphs, save_info, load_info, generate_mask_tensor, deprecate_property
import zipfile

backend = os.environ.get('DGLBACKEND', 'pytorch')


def _pickle_load(pkl_file):
    if sys.version_info > (3, 0):
        return pkl.load(pkl_file, encoding='latin1')
    else:
        return pkl.load(pkl_file)


def _read_sparse_features(feature_path):
    """
    Reading the feature matrix stored as JSON from the disk.
    :param feature_path: Path to the JSON file.
    :return features: Feature sparse COO matrix.
    """
    features = json.load(open(feature_path))
    index_1 = [int(k) for k, v in features.items() for fet in v]
    index_2 = [int(fet) for k, v in features.items() for fet in v]
    values = [1.0]*len(index_1)
    nodes = [int(k) for k, v in features.items()]
    node_count = max(nodes)+1
    feature_count = max(index_2)+1
    features = sp.csr_matrix(sp.coo_matrix((values, (index_1, index_2)),
                                                   shape=(node_count, feature_count),
                                                   dtype=np.float32))
    return features


def _read_csv(filepath):
    content = []
    flag = 1
    with open(filepath, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if flag:
                flag = 0
                continue
            content.append(tuple([int(i) for i in row]))
    return content



def _read_csv_twitch(filepath):
    content = []
    flag = 1
    with open(filepath, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            if flag:
                flag = 0
                continue
            if row[4] == 'False':
                content.append(0)
            elif row[4] == 'True':
                content.append(1)
    return content


def _read_external_dataset(root):
    for filename in os.listdir(root):
        filepath = os.path.join(root,filename)
        print(filename)
        if filepath.endswith('target.csv'):
            labels = _read_csv_twitch(filepath)           
        elif filepath.endswith('edges.csv'):
            edges_list = _read_csv(filepath)
        elif filepath.endswith('.json'):
            features = _read_sparse_features(filepath)
    return features, edges_list, labels


class ExternalDataset(DGLDataset):
    r"""The Basic DGL Builtin Dataset.

    Parameters
    ----------
    name : str
        Name of the dataset.
    url : str
        Url to download the raw dataset.
    raw_dir : str
        Specifying the directory that will store the
        downloaded data or the directory that
        already stores the input data.
        Default: ~/.dgl/
    hash_key : tuple
        A tuple of values as the input for the hash function.
        Users can distinguish instances (and their caches on the disk)
        from the same dataset class by comparing the hash values.
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: False
    """

    def __init__(self, name, urls, raw_dir=None, hash_key=(), force_reload=False, verbose=False):

        self._urls = urls
        super().__init__(name,
                        url=None,
                        raw_dir=raw_dir,
                        save_dir=None,
                        hash_key=hash_key,
                        force_reload=force_reload,
                        verbose=verbose)

    def download(self):
        r""" Automatically download data and extract it.
        """
        
        file_path = os.path.join(self.raw_dir, 'twitch.zip')
        download(self.urls, path=file_path)
        zip_file = zipfile.ZipFile(file_path)
        zip_file.extractall(self.raw_dir)
        

    @property
    def urls(self):
        r"""Get url to download the raw dataset.
        """
        return self._urls

    @property
    def raw_path(self):
        r"""Directory contains the input data files.
            By default raw_path = os.path.join(self.raw_dir, self.name)
        """
        return os.path.join(self.raw_dir, 'twitch', self.name)


class SocialNetworkDataset(ExternalDataset):

    r"""The social network graph dataset, including last.fm, github, deezer_europe and facebook.
    Nodes mean users and edges mean users' relationship.

    Parameters
    -----------
    name: str
      name can be 'twitch_[country]'
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.
    """
    

    def __init__(self, name, raw_dir=None, force_reload=False, verbose=True):
        
        raw_dir = '%s/.dgl/' % (os.path.expanduser('~'))
        urls = 'https://snap.stanford.edu/data/twitch.zip'
        super().__init__(name,
                        urls=urls,
                        raw_dir=raw_dir,
                        force_reload=force_reload,
                        verbose=verbose)
        
    def process(self):
        """
        Loads input data from data directory and Processes them
        """
        root = self.raw_path
        features, edges_list, labels = _read_external_dataset(root)
        num_classes = max(labels) + 1
        graph = nx.DiGraph(edges_list)
        features = features.tolil()
        labels = np.array(labels)

        decile_size = labels.shape[0] // 10
        idx_test = range(decile_size * 3)
        idx_train = range(decile_size * 3, decile_size * 9)
        idx_val = range(decile_size * 9, decile_size * 10)

        train_mask = _sample_mask(idx_train, labels.shape[0])
        val_mask = _sample_mask(idx_val, labels.shape[0])
        test_mask = _sample_mask(idx_test, labels.shape[0])

        self._graph = graph
        g = from_networkx(graph)

        g.ndata['train_mask'] = generate_mask_tensor(train_mask)
        g.ndata['val_mask'] = generate_mask_tensor(val_mask)
        g.ndata['test_mask'] = generate_mask_tensor(test_mask)
        g.ndata['label'] = F.tensor(labels)
        g.ndata['feat'] = F.tensor(_preprocess_features(features), dtype=F.data_type_dict['float32'])
        self._num_classes = num_classes
        self._labels = labels
        self._g = g

        if self.verbose:
            print('Finished data loading and preprocessing.')
            print('  NumNodes: {}'.format(self._g.number_of_nodes()))
            print('  NumEdges: {}'.format(self._g.number_of_edges()))
            print('  NumFeats: {}'.format(self._g.ndata['feat'].shape[1]))
            print('  NumClasses: {}'.format(self.num_classes))
            print('  NumTrainingSamples: {}'.format(
                F.nonzero_1d(self._g.ndata['train_mask']).shape[0]))
            print('  NumValidationSamples: {}'.format(
                F.nonzero_1d(self._g.ndata['val_mask']).shape[0]))
            print('  NumTestSamples: {}'.format(
                F.nonzero_1d(self._g.ndata['test_mask']).shape[0]))

    def has_cache(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        if os.path.exists(graph_path) and \
                os.path.exists(info_path):
            return True

        return False

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        save_graphs(str(graph_path), self._g)
        save_info(str(info_path), {'num_classes': self.num_classes})

    def load(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        graphs, _ = load_graphs(str(graph_path))

        info = load_info(str(info_path))
        self._g = graphs[0]
        # graph = graph.clone()
        # graph.pop('train_mask')
        # graph.pop('val_mask')
        # graph.pop('test_mask')
        # graph.pop('feat')
        # graph.pop('label')
        # graph = to_networkx(graph)
        # self._graph = nx.DiGraph(graph)

        self._num_classes = info['num_classes']
        self._g.ndata['train_mask'] = generate_mask_tensor(self._g.ndata['train_mask'].numpy())
        self._g.ndata['val_mask'] = generate_mask_tensor(self._g.ndata['val_mask'].numpy())
        self._g.ndata['test_mask'] = generate_mask_tensor(self._g.ndata['test_mask'].numpy())
        # hack for mxnet compatability

        if self.verbose:
            print('  Twitch Dataset: {}'.format(self.name))
            print('  NumNodes: {}'.format(self._g.number_of_nodes()))
            print('  NumEdges: {}'.format(self._g.number_of_edges()))
            print('  NumFeats: {}'.format(self._g.ndata['feat'].shape[1]))
            print('  NumClasses: {}'.format(self.num_classes))
            print('  NumTrainingSamples: {}'.format(
                F.nonzero_1d(self._g.ndata['train_mask']).shape[0]))
            print('  NumValidationSamples: {}'.format(
                F.nonzero_1d(self._g.ndata['val_mask']).shape[0]))
            print('  NumTestSamples: {}'.format(
                F.nonzero_1d(self._g.ndata['test_mask']).shape[0]))

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1

    @property
    def save_name(self):
        return self.name + '_dgl_graph'

    @property
    def num_labels(self):
        deprecate_property('dataset.num_labels', 'dataset.num_classes')
        return self.num_classes

    @property
    def num_classes(self):
        return self._num_classes

    """ Social Network graph is used in many examples
        We preserve these properties for compatability.
    """

    @property
    def graph(self):
        deprecate_property('dataset.graph', 'dataset[0]')
        return self._graph

    @property
    def train_mask(self):
        deprecate_property('dataset.train_mask', 'g.ndata[\'train_mask\']')
        return F.asnumpy(self._g.ndata['train_mask'])

    @property
    def val_mask(self):
        deprecate_property('dataset.val_mask', 'g.ndata[\'val_mask\']')
        return F.asnumpy(self._g.ndata['val_mask'])

    @property
    def test_mask(self):
        deprecate_property('dataset.test_mask', 'g.ndata[\'test_mask\']')
        return F.asnumpy(self._g.ndata['test_mask'])

    @property
    def labels(self):
        deprecate_property('dataset.label', 'g.ndata[\'label\']')
        return F.asnumpy(self._g.ndata['label'])

    @property
    def features(self):
        deprecate_property('dataset.feat', 'g.ndata[\'feat\']')
        return self._g.ndata['feat']


def _preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.asarray(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return np.asarray(features.todense())


def _parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def _sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return mask
