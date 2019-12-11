"""
From Elisa's code base: https://raw.githubusercontent.com/elisaF/typing-classification/master/helper_functions.py
"""


from __future__ import division
import codecs
import csv
import logging
import unicodedata
import pickle

logger = logging.getLogger("helper_functions")
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler("helper_functions.log")
logger.addHandler(fh)

lm_ngram1 = {}
lm_ngram2 = {}
lm_ngram3 = {}
lm_ngram4 = {}
lm_ngram5 = {}
dict_lm_objects = {1: lm_ngram1, 2: lm_ngram2, 3: lm_ngram3, 4: lm_ngram4, 5: lm_ngram5}


def has_diacritic(input_str):
    len_word = len(input_str)
    len_nfkd_form = len(unicodedata.normalize("NFKD", input_str))
    return len_word != len_nfkd_form


def normalize(char):
    if has_diacritic(char):
        return unicodedata.normalize("NFKD", char)[0]
    else:
        return char


def transform_for_lm(string):
    new_string = ""
    for index, char in enumerate(string):
        if char == " ":
            new_string = new_string + u"}"
        elif (char == "." and len(string) == (index + 1)) or (
            char == "." and string[index + 1] == " "
        ):  # treat period as end of sentence
            new_string = new_string + u"</s>"
        else:
            new_string = new_string + char.lower()
        if len(string) > (index + 1):
            new_string = new_string + u" "
    new_string = new_string.casefold()
    return new_string


def get_prob_chars(string, lm_file):
    build_prob_dicts(lm_file)
    use_backoff_weight = False
    length = len(string)
    string_lm = transform_for_lm(string)
    logger.debug("Original string: " + string + ", transformed string: " + string_lm)
    old_string_lm = string_lm
    while string_lm not in dict_lm_objects[length]:
        use_backoff_weight = True
        old_string_lm = string_lm
        string_lm = backoff(string_lm)

        logger.debug("Char sequence after backoff is: " + string_lm)
        length = length - 1
        logger.debug("Length is now: " + str(length))
        if length == 0:
            break

    if length == 0:
        prob = "NA"
        logger.debug("Prob is: " + prob)
        use_backoff_weight = False
    else:
        ngram_values = dict_lm_objects[length][string_lm]
        prob = ngram_values[0]
        logger.debug("Prob is: " + str(prob))
    if use_backoff_weight:
        backoff_string = old_string_lm[: (length * 2) - 1]  # account for blank spaces
        logger.debug("Backoff_string is " + backoff_string)
        backoff_length = int((len(backoff_string) + 1) / 2)  # account for blank spaces
        logger.debug("Length of backoff string is " + str(backoff_length))
        if backoff_string not in dict_lm_objects[backoff_length]:
            backoff_weight = 0
        else:
            ngram_values_backoff = dict_lm_objects[backoff_length][backoff_string]
            if len(ngram_values_backoff) == 1:
                backoff_weight = 0
            else:
                backoff_weight = ngram_values_backoff[1]
        logger.debug("Backoff weight is " + str(backoff_weight))
        return float(prob) + float(backoff_weight)
    else:
        return prob


def backoff(string):
    if string.startswith("</s>") or string.startswith("<s>"):
        return string[5:]
    else:
        return string[2:]  # drop 1st character and blank space


def build_prob_dicts(lm_file):
    global lm_ngram1
    if not lm_ngram1:
        # find correct n-gram section
        ngram_match = "-grams:"
        index = 0
        with open(lm_file) as f:
            for line in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                if len(line) == 0:
                    continue
                elif line[0].endswith(ngram_match):
                    index += 1
                elif len(line) < 2:
                    continue
                else:
                    key = line[1]
                    if len(line) == 2:  # sometimes we don't have backoff weight
                        dict_lm_objects[index][key] = [line[0]]
                    else:
                        dict_lm_objects[index][key] = [line[0], line[2]]


def create_vocab_for_lm(pickle_file_1, pickle_file_2=None):
    outputfile_name = "all_chars.txt"
    df_1 = pickle.load(open(pickle_file_1, "rb"))
    context_1 = df_1["Error Context"].str.replace(" ", "}")  # convert spaces to }
    all_chars = set(list(" ".join(list(context_1.values))))

    if pickle_file_2:
        df_2 = pickle.load(open(pickle_file_2, "rb"))
        context_2 = df_2["Error Context"].str.replace(" ", "}")  # convert spaces to }
        chars_2 = set(list(" ".join(list(context_2.values)).casefold()))  # casefold
        all_chars = chars_2.union(all_chars)

    with codecs.open(outputfile_name, "w", "utf-8") as outfile:
        for char in all_chars:
            outfile.write(char + "\n")


def casefold_file(file_name):
    casefolded_file_name = file_name + "_casefolded"
    with codecs.open(file_name, "r", "utf-8") as infile:
        with codecs.open(casefolded_file_name, "w", "utf-8") as outfile:
            for line in infile:
                outfile.write(line.casefold())
