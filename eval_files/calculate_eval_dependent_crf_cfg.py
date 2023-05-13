"""
Kalkulasi performa dari algoritma CRF dan CFG secara 
terintegrasi (hasil dari prediksi CRF akan digunakan 
pada CFG).
"""
from sklearn import metrics
import matplotlib.pyplot as plt
import os

curr_dir = os.path.dirname(__file__)
