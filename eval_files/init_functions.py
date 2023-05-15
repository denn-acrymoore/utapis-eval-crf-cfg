import os
import re
import nltk
from nltk.tag import CRFTagger
from nltk.parse.chart import (
    SteppingChartParser,
    FilteredSingleEdgeFundamentalRule,
    LeafInitRule,
    FilteredBottomUpPredictCombineRule,
)
import unicodedata
import json


def initialize_crf_cfg():
    """NOTE:
    - os.getcwd()   --> Print working directory (not suitable for modules).
    - __file__      --> Print absolute path for this file "main.py" (suitable
                        for modules).
    """
    # print("os.getcwd(): " + os.getcwd())
    # print("__file__: " + __file__)
    model_path = os.path.dirname(__file__)
    # print("MODEL PATH:", model_path)

    # Initialize CFG:
    grammar = nltk.data.load(
        "file:" + os.path.join(model_path, "utapis_sintaksis_kalimat_v2_skripsi.cfg"),
        "cfg",
    )

    LEFT_CORNER_STRATEGY = [
        LeafInitRule(),
        FilteredBottomUpPredictCombineRule(),
        FilteredSingleEdgeFundamentalRule(),
    ]

    utapis_scp = SteppingChartParser(
        grammar=grammar, strategy=LEFT_CORNER_STRATEGY, trace=0
    )

    # Initialize CRF:
    def custom_feature_func(tokens, idx):
        """
        Features to extract:
        1.  Current word.
        2.  Previous word (if any).
        3.  Previous previous word (if any).
        4.  Next word (if any).
        5.  Next next word (if any).
        6.  Is the word capitalized?
        7.  Is the first word in the sentence?
        8.  Does it contain punctuation?
        9.  Does it contain a number?
        10. Is the word all number (with or without ., between number)?
        11. Is the word all uppercase?
        12. Is the word all uppercase + symbol?
        13. Prefixes up to length 4.
        14. Suffixes up to length 4.

        :return: a list which contains the features
        :rtype: list(str)
        """
        token = tokens[idx]
        feature_list = []

        # Check if token out-of-range
        if not token:
            return feature_list

        # Feature 1: Current word
        feature_list.append("WORD_" + token)

        # Feature 2: Previous word (if any)
        if idx > 0:
            feature_list.append("PREV_WORD_" + tokens[idx - 1])

        # Feature 3: Previous previous word (if any)
        if idx > 1:
            feature_list.append("PREV_PREV_WORD_" + tokens[idx - 2])

        # Feature 4: Next word (if any)
        if idx < len(tokens) - 1:
            feature_list.append("NEXT_WORD_" + tokens[idx + 1])

        # Feature 5: Next next word (if any)
        if idx < len(tokens) - 2:
            feature_list.append("NEXT_NEXT_WORD_" + tokens[idx + 2])

        # Feature 6: Is the word capitalized?
        if token[0].isupper():
            feature_list.append("CAPITALIZATION")

        # Feature 7: Is the first word in the sentence?
        if idx == 0:
            feature_list.append("FIRST_WORD")

        # Feature 8: Does it contain punctuation?
        punc_cat = {"Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"}
        if all(unicodedata.category(x) in punc_cat for x in token):
            feature_list.append("PUNCTUATION")

        # Feature 9: Does it contain a number?
        if re.search(r"\d", token) is not None:
            feature_list.append("HAS_NUM")

        # Feature 10: Is the word all number (with or without ., between number)?
        if re.search(r"^\d+([.,]\d+)*$", token) is not None:
            feature_list.append("ALL_NUM")

        # Feature 11: Is the word all uppercase?
        if re.search(r"^[A-Z]+$", token) is not None:
            feature_list.append("UPPERCASE")

        # Feature 12: Is the word all uppercase + symbol?
        if re.search(r"^[A-Z@#$%^&*]+$", token) is not None:
            feature_list.append("UPPERCASE_SYMBOL")

        # Feature 13: Prefixes up to length 4
        if len(token) > 1:
            feature_list.append("PREF_" + token[:1])
        if len(token) > 2:
            feature_list.append("PREF_" + token[:2])
        if len(token) > 3:
            feature_list.append("PREF_" + token[:3])
        if len(token) > 4:
            feature_list.append("PREF_" + token[:4])

        # Feature 14: Suffixes up to length 4
        if len(token) > 1:
            feature_list.append("SUF_" + token[-1:])
        if len(token) > 2:
            feature_list.append("SUF_" + token[-2:])
        if len(token) > 3:
            feature_list.append("SUF_" + token[-3:])
        if len(token) > 4:
            feature_list.append("SUF_" + token[-4:])

        return feature_list

    utapis_crf_tagger = CRFTagger(feature_func=custom_feature_func)
    utapis_crf_tagger.set_model_file(
        os.path.join(model_path, "utapis_crf_model_skripsi.crf.tagger")
    )

    return utapis_crf_tagger, utapis_scp


def stepping_chart_parsing(scp, tags):
    scp.initialize(tags)

    for step in scp.step():
        # Berhenti bila ditemukan parsing yang lengkap.
        if len(list(scp.parses())) > 0:
            break

        # Berhenti bila sudah tidak ada lagi kemungkinan parsing
        # yang bisa ditambahkan.
        if step is None:
            break

    # Return generator.
    return scp.parses()


def get_crf_results(crf_tagger, preprocessed_sentence_list):
    """
    :return: list of tagged sentences.
    :rtype: list(list(tuple(str, str)))
    """
    return crf_tagger.tag_sents(preprocessed_sentence_list)


def get_cfg_results(scp, list_of_tags):
    """
    scp: SteppingChartParser object
    list_of_tags: list(list(str))

    Return: cfg_results (list(bool))
    """
    cfg_results = []
    for idx, tags in enumerate(list_of_tags):
        print(f"Kalimat {idx+1}/{len(list_of_tags)}")
        generator = stepping_chart_parsing(scp, tags)
        generator_content_count = len(list(generator))

        # print(f"generator_content_count = {generator_content_count}", end=" ")

        if generator_content_count <= 0:
            cfg_results.append(False)
        elif generator_content_count > 0:
            cfg_results.append(True)
    return cfg_results


def get_crf_cfg_actual_values():
    """
    @return crf_all_data: list(list(tuple(str, str)))
    @return cfg_all_data: list(tuple(boolean, list(str)))
    """
    curr_dir = os.path.dirname(__file__)
    eval_path = os.path.join(curr_dir, os.pardir, "processed and manually tagged")

    # List semua tagged file.
    file_list = os.listdir(eval_path)

    # Ambil file-file yang memiliki extension .json.
    eval_list = []
    for data in file_list:
        data_path = os.path.join(eval_path, data)
        if os.path.isfile(data_path) and os.path.splitext(data_path)[1] == ".json":
            eval_list.append(data_path)

    eval_list.sort()

    # Baca setiap file dan ambil semua actual values dari CRF dan CFG.
    # NOTE: data semua kalimat di tiap file digabung menjadi satu list.
    crf_all_data = []
    cfg_all_data = []

    for eval_file_path in eval_list:
        temp_crf_data = []
        temp_cfg_data = []

        with open(eval_file_path, "r") as fp:
            json_obj = json.load(fp)

        for sentence_obj in json_obj:
            tag_tuple_format = []
            tag_only = []

            for tag in sentence_obj[1]:
                tag_tuple_format.append((tag[0], tag[1]))
                tag_only.append(tag[1])
            temp_crf_data.append(tag_tuple_format)
            temp_cfg_data.append((sentence_obj[0], tag_only))

        crf_all_data.append(temp_crf_data)
        cfg_all_data.append(temp_cfg_data)

    return crf_all_data, cfg_all_data


if __name__ == "__main__":
    crf_all_data, cfg_all_data = get_crf_cfg_actual_values()
    print(len(crf_all_data))
    print(len(cfg_all_data))
    print(crf_all_data[0])
    print()
    print(cfg_all_data[0])
    print()
