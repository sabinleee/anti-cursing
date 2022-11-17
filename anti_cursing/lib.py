import warnings 
warnings.filterwarnings(action= 'ignore')

import os
from itertools import combinations
import random
import logging
from collections import OrderedDict

from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity,euclidean_distances,pairwise_distances
from sklearn.metrics import precision_recall_fscore_support, accuracy_score