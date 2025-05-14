import os
import csv
import re
import matplotlib.pyplot as plt
from SaGe_main.src.sage_tokenizer import *
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece, WordLevel
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def analyze_tokenization(tokenizers_list, ff_data, l1, l2, algo, dir):
    """
    This function computes an analysis on how different tokenizers split False Friends words
    :param tokenizers_list: a list of tokenizers, [l1 tokenizer, l2 tokenizer, l1_l2 tokenizer]
    :param ff_data: the false friends data
    :param l1: the first language
    :param l2: the second language
    :param algo: the name of the algorithm
    :param dir: the directory to save the results figure
    :return:
    """
    # init cases with value 0
    tokenization_cases = ["same_splits", f"{l1}_t==multi_t", f"{l2}_t==multi_t", "different_splits", f"{l1}_t=={l2}_t"]
    num_tokens_diff = dict.fromkeys(tokenization_cases, 0)
    for ff in ff_data:
        word_tokenization = []
        num_tokens = []
        for t in tokenizers_list:
            if "SAGE" in algo:
                res = t.tokenize_to_encoded_str(ff)
            else:
                res = t.encode(ff).tokens
            word_tokenization.append(res)
            num_tokens.append(len(res))
        # Same splits throughout all tokenizers
        if word_tokenization[0] == word_tokenization[2] and word_tokenization[1] == word_tokenization[2]:
            num_tokens_diff["same_splits"] += 1
        # Same tokenization between language1 and multilingual tokenizer
        elif word_tokenization[0] == word_tokenization[2]:
            num_tokens_diff[f"{l1}_t==multi_t"] += 1
        # Same tokenization between language2 and multilingual tokenizer
        elif word_tokenization[1] == word_tokenization[2]:
            num_tokens_diff[f"{l2}_t==multi_t"] += 1
        # All different tokenization
        elif word_tokenization[0] != word_tokenization[1] and word_tokenization[0] != word_tokenization[2] and \
                word_tokenization[1] != word_tokenization[2]:
            num_tokens_diff["different_splits"] += 1
        # Same tokenization between language1 and langauge2, but different from Multi tokenizer
        elif word_tokenization[0] == word_tokenization[1]:
            num_tokens_diff[f"{l1}_t=={l2}_t"] += 1
    
    num_words = len(ff_data)
    fig_save_path = f"{dir}/tokenization_cases_{l1}_{l2}_{algo}.png"
    title = f"Tokenization Cases:{l1}, {l2}\nAlgo:{algo}\nNum ff: {num_words}"
    
    plt.figure(figsize=(8, 10))
    x_axis = list(num_tokens_diff.keys())
    y_axis = [v for v in list(num_tokens_diff.values())]
    plt.bar(x_axis, y_axis)
    plt.xticks(rotation=30, fontsize=12)
    plt.xlabel("Tokenization Splits")
    plt.ylabel("Amount of Tokenization Case")
    plt.title(title, fontsize=15)
    plt.savefig(fig_save_path)
    plt.show()

def analyze_same_words_same_splits():
    pass
def write_tokenization_split(tokenizers, ff_data, l1, l2, algo, dir):
    """
    Writes the tokenization splits of different tokenizers to a .txt file
    :param tokenizers: a list of tokenizers, [l1 tokenizer, l2 tokenizer, l1_l2 tokenizer]
    :param ff_data: the ff data
    :param l1: language 1 (english)
    :param l2: language 2
    :param algo: the algorithm used
    :param dir: path to save .txt file
    :return:
    """
    with open(dir, 'w', encoding='utf-8') as f:
        f.write(f"{l1}_tokenizer, {l2}_tokenizer, {l1}_{l2}_tokenizer\n")
        for ff in ff_data:
            to_write = f""
            for t in tokenizers:
                if "SAGE" in algo:
                    to_write += f"{t.tokenize_to_encoded_str(ff)}"
                else:
                    to_write += f"{t.encode(ff).tokens}"
            to_write += "\n"
            f.write(to_write)
            
def get_same_words_across_languages(languages_set):
    """
    This function returns all the words that are written exactly the same as English
    :param languages_set: the language set
    :return: a dictionary --> {language : set(ff_words)}
    """
    language_dict = dict()
    same_words_eng_l2 = dict()
    for l in languages_set:
        with open(f"./all_words_in_all_languages/{l}/{l}.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()[0].strip().lower().split(",")
        language_dict[l] = set(lines)
    
    for l in languages_set[1:]:
        ff_words = language_dict["English"] & language_dict[l]
        same_words_eng_l2[l] = ff_words
    return same_words_eng_l2

def get_word_frequencies_training_corpus(dir):
    """
    Get the word frequencies of words for each language in the directory. Looks at all words as lower case, so the word
    "a" and "A" are considered the same
    :param dir: the directory
    :return: dictionary --> {language : {word: word_frequency}}
    """
    word_frequencies = dict()
    for d in os.listdir(dir):
        word_frequencies[d] = dict()
        for path in os.listdir(f"{dir}/{d}"):
            with open(f"{dir}/{d}/{path}", 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                word, freq = line.split("\t")[1:]
                # only lower case words
                word = word.lower()
                if word in word_frequencies[d]:
                    word_frequencies[d][word] = word_frequencies[d][word] + int(freq.strip())
                else:
                    word_frequencies[d][word] = int(freq.strip())
    return word_frequencies
    

def load_tokenizer(dir):
    """
    This function loads a tokenizer, a Sentenpiece tokenizer or a SaGe tokenizer
    :param dir: the directory to load the tokenizer from
    :return: a trained tokenizer
    """
    if dir.endswith(".json"):
        return Tokenizer.from_file(dir)
    else:
        with open(dir, "r") as f:
            sage_vocab = [bytes.fromhex(line.strip()) for line in f]
        return SaGeTokenizer(initial_vocabulary=sage_vocab)


def extract_lang_and_tokenizer(filename):
    """
    Extracts the language and tokenizer type from a filename.
    Examples:
        'de_BPE.json'              -> ['de', 'BPE']
        'en_de_BPE.json'           -> ['en_de', 'BPE']
        'en_de_BPE_SAGE.vocab'     -> ['en_de', 'BPE_SAGE']
        'de_UNI_SAGE.vocab'        -> ['de', 'UNI_SAGE']

    :param filename: The filename to parse
    :return: [language, tokenizer_type]
    """
    base = os.path.splitext(filename)[0]
    parts = base.split('_')
    
    # Determine language and tokenizer type
    if len(parts) == 2:
        lang = parts[0]
        tokenizer = parts[1]
    else:
        lang = '_'.join(parts[:-2]) if parts[-2] in {"BPE", "UNI"} else '_'.join(parts[:-1])
        tokenizer = '_'.join(parts[len(lang.split('_')):])
    return [lang, tokenizer]



languages_set = ["English", "French", "Spanish", "German", "Swedish", "Italian", "Romanian"]
# Get ff_data and put in dictionary {language: {ff_words}}
with open("./args_script.txt", 'r', encoding='utf-8') as f:
    experiments = f.readlines()
ff_data = dict()
for e in experiments:
    language, training_data_path, ff_data_path =  e.split(",")
    ff_data_path = ff_data_path.replace("\\", "/").strip()
    with open(ff_data_path, 'r', encoding='utf-8') as f:
        ff_data[language] = list(csv.DictReader(f))
for k, v in ff_data.items():
    ff_words = set()
    for i in range(len(v)):
        ff_words.add(v[i]["False Friend"])
    ff_data[k] = ff_words

# ff_data[language] holds a set of ff words for that specific language
# same_words[language] holds a set of words from language "language" that are written exactly the same as English
# word_freq[language] holds a dictionary for every language. In that dictionary, each item is a word from the corpus with it's number of appearances, i.e. frequency
same_words = get_same_words_across_languages(languages_set)
word_freq = get_word_frequencies_training_corpus("./training_data/words")
tokenizer_dict = dict()

vocab_size = 3000
l1 = "en"
l1_tokenizers = dict()
for token_file in os.listdir(f"./experiments/{vocab_size}/{l1}"):
    file_name, file_type = token_file.split(".")
    algo = file_name.split("_", 1)[1]
    l1_tokenizers[algo] = load_tokenizer(f"./experiments/{vocab_size}/{l1}/{token_file}")

for l1_l2 in os.listdir(f"./experiments/{vocab_size}"):
    # skip the en directory
    if l1_l2 == l1:
        continue
    l2 = l1_l2.split("_", 1)[1]
    for cur_algo, l1_tokenizer in l1_tokenizers.items():
        l2_tokenizer = None
        l1_l2_tokenizer = None
        for token_file in os.listdir(f"./experiments/{vocab_size}/{l1_l2}"):
            l3, algo_name = extract_lang_and_tokenizer(token_file)
            if algo_name == cur_algo:
                some_t = load_tokenizer(f"./experiments/{vocab_size}/{l1_l2}/{token_file}")
                if l3 == l1_l2:
                    l1_l2_tokenizer = some_t
                else:
                    l2_tokenizer = some_t
        analyze_tokenization([l1_tokenizer, l2_tokenizer, l1_l2_tokenizer], ff_data[l2], l1, l2, cur_algo,
                             f"./analysis/{vocab_size}/{l1_l2}/graphs")
        write_tokenization_split([l1_tokenizer, l2_tokenizer, l1_l2_tokenizer], ff_data[l2], l1, l2, cur_algo,
                                 f"./analysis/{vocab_size}/{l1_l2}/tokenization/{cur_algo}.txt")
        

        






