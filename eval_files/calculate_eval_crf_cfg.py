"""
Kalkulasi performa dari algoritma CRF dan CFG (CRF dan CFG 
secara individual dan saat terintegrasi bersama).
"""
from sklearn import metrics
import matplotlib.pyplot as plt
import os
from init_functions import (
    initialize_crf_cfg,
    get_crf_results,
    get_cfg_results,
    get_crf_cfg_actual_values,
)
from datetime import datetime
import pytz
import numpy as np

now = datetime.now(tz=pytz.timezone("Asia/Jakarta"))

curr_dir = os.path.dirname(__file__)
fig_crf_only = os.path.join(
    curr_dir, f"({now.strftime('%Y-%b-%d_%H-%M-%S')})_fig_crf_only.png"
)
fig_cfg_only = os.path.join(
    curr_dir, f"({now.strftime('%Y-%b-%d_%H-%M-%S')})_fig_cfg_only.png"
)
fig_crf_cfg = os.path.join(
    curr_dir, f"({now.strftime('%Y-%b-%d_%H-%M-%S')})_fig_crf_cfg.png"
)

file_name = os.path.join(
    curr_dir, f"({now.strftime('%Y-%b-%d_%H-%M-%S')})_eval_result_crf_cfg.txt"
)

# List dari label-label CRF:
crf_labels = [
    "<verba>",
    "<preposisi>",
    "<dengan>",
    "<bukan>",
    "<entah>",
    "<jangankan>",
    "<pun>",
    "<nomina>",
    "<artikel>",
    "<adjektiva>",
    "<adverbia>",
    "<numeralia>",
    "<kj_koor_tak_hingga>",
    "<kj_koor_terhingga>",
    "<kj_koor_terhingga_klausa>",
    "<kj_sub_tnp_koma_kt_pertama>",
    "<kj_sub_tnp_koma_kt_kedua>",
    "<kj_sub_dgn_koma>",
    "<kj_sub_yang>",
    "<kj_sub_bahwa>",
    "<kj_sub_berbeda>",
    "<kj_atr_pertama>",
    "<kj_atr_kedua>",
    "<kj_atr_ketiga>",
    "<kj_atr_adapun>",
    "<td_koma>",
    "<td_tengah_kal>",
    "<td_akhir_kal>",
    "<kurung_buka>",
    "<kurung_tutup>",
    "<kutip_awal>",
    "<kutip_akhir>",
    "<simbol>",
    "<tidak>",
    "<per>",
    "<belum>",
    "<demi>",
]

fp = open(file_name, "w")
utapis_crf_tagger, utapis_scp = initialize_crf_cfg()

# Ambil semua data CRF dan CFG dari tiap artikel.
crf_all_data, cfg_all_data = get_crf_cfg_actual_values()

# Ambil 'predicted values' dan 'actual values'.
crf_predicted_data = []

crf_actual_data_1d = []
cfg_actual_data_1d = []

crf_predicted_data_1d = []
cfg_predicted_data_1d = []

# Ambil predicted value dan actual value CRF
for idx, crf_data_per_article in enumerate(crf_all_data):
    print(f"CRF: Processing Article {idx+1}")
    crf_article_word_only = []
    for sentence in crf_data_per_article:
        crf_article_word_only.append([item[0] for item in sentence])
        crf_actual_data_1d += [item[1] for item in sentence]

    result = get_crf_results(utapis_crf_tagger, crf_article_word_only)
    crf_predicted_data.append(result)

    for r in result:
        crf_predicted_data_1d += [x[1] for x in r]

print()

# Evaluasi algoritma CRF.
fp.write("======= CRF Only Evaluation =======\n")
print("======= CRF Only Evaluation =======")
cm = metrics.confusion_matrix(
    crf_actual_data_1d, crf_predicted_data_1d, labels=crf_labels
)
np.savetxt(fp, cm, fmt="%d")
fp.write("\n\n")

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=crf_labels
)

cm_display.plot()
plt.savefig(fig_crf_only)
print("CRF Confusion Matrix Pic Generated!")

classification_report = metrics.classification_report(
    crf_actual_data_1d, crf_predicted_data_1d, labels=crf_labels
)
fp.write(classification_report)
fp.write("\n\n")
print("CRF Classification Report Written!")

print()

# Ambil predicted value dan actual value CFG
for idx, cfg_data_per_article in enumerate(cfg_all_data):
    print(f"CFG: Processing Article {idx+1}")
    cfg_article_tag_only = [item[1] for item in cfg_data_per_article]
    cfg_actual_data_1d += [item[0] for item in cfg_data_per_article]

    result = get_cfg_results(utapis_scp, cfg_article_tag_only)
    cfg_predicted_data_1d += result

print()

# Evaluasi algoritma CFG.
fp.write("======= CFG Only Evaluation =======\n")
print("======= CFG Only Evaluation =======")
cm = metrics.confusion_matrix(
    cfg_actual_data_1d, cfg_predicted_data_1d, labels=[True, False]
)
np.savetxt(fp, cm, fmt="%d")
fp.write("\n\n")

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=[True, False]
)

cm_display.plot()
plt.savefig(fig_cfg_only)
print("CFG Confusion Matrix Pic Generated!")

classification_report = metrics.classification_report(
    cfg_actual_data_1d, cfg_predicted_data_1d, labels=[True, False]
)
fp.write(classification_report)
fp.write("\n\n")
print("CFG Classification Report Written!")


fp.write("======= CRF + CFG Evaluation =======\n")
print("======= CRF + CFG Evaluation =======")
# Hitung ulang CFG berdasarkan hasil prediksi CRF
cfg_predicted_data_1d = []

for idx, crf_result_per_article in enumerate(crf_predicted_data):
    print(f"CFG: Processing Article {idx+1}")
    tag_only = []
    for sentence in crf_result_per_article:
        tag_only.append([item[1] for item in sentence])

    result = get_cfg_results(utapis_scp, tag_only)
    cfg_predicted_data_1d += result

cm = metrics.confusion_matrix(
    cfg_actual_data_1d, cfg_predicted_data_1d, labels=[True, False]
)
np.savetxt(fp, cm, fmt="%d")
fp.write("\n\n")

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=[True, False]
)

cm_display.plot()
plt.savefig(fig_crf_cfg)
print("CRF + CFG Confusion Matrix Pic Generated!")

classification_report = metrics.classification_report(
    cfg_actual_data_1d, cfg_predicted_data_1d, labels=[True, False]
)
fp.write(classification_report)
fp.write("\n\n")
print("CRF + CFG Classification Report Written!")


fp.close()
