import os
import csv
import re
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
from SaGe_main.src.sage_tokenizer import *
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece, WordLevel
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

def analyze_tokenization(tokenizers_list, word_list, l1, l2, algo):
    """
    This function computes an analysis on how different tokenizers split words
    :param tokenizers_list: a list of tokenizers, [l1 tokenizer, l2 tokenizer, l1_l2 tokenizer]
    :param word_list: list of words
    :param l1: the first language
    :param l2: the second language
    :param algo: the name of the algorithm
    :return:
    """
    # init cases with value 0
    tokenization_cases = ["same_splits", f"{l1}_t==multi_t", f"{l2}_t==multi_t", "different_splits", f"{l1}_t=={l2}_t"]
    num_tokens_diff = {k: 0 for k in tokenization_cases}
    
    for word in word_list:
        word_tokenization = []
        num_tokens = []
        for t in tokenizers_list:
            if "SAGE" in algo:
                res = t.tokenize_to_encoded_str(word)
            else:
                res = t.encode(word).tokens
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
    
    return num_tokens_diff
    
    
def intrinsic_analysis(tokenizers_list, ff_data, word_frequencies, algo, l1, l2, dir):
    tokenization_cases = ["same_splits", f"{l1}_t==multi_t", f"{l2}_t==multi_t", "different_splits", f"{l1}_t=={l2}_t"]
    case_word_dict = {k: [] for k in tokenization_cases}
    case_num_tokens_dict = {k: [] for k in tokenization_cases}
    
    for word in ff_data:
        word_tokenization = []
        num_tokens = []
        for t in tokenizers_list:
            if "SAGE" in algo:
                res = t.tokenize_to_encoded_str(word)
            else:
                res = t.encode(word).tokens
            word_tokenization.append(res)
            num_tokens.append(len(res))
        # Same splits throughout all tokenizers
        if word_tokenization[0] == word_tokenization[2] and word_tokenization[1] == word_tokenization[2]:
            case_word_dict["same_splits"].append(word)
            case_num_tokens_dict["same_splits"].append([word, len(word_tokenization[0])])
        # Same tokenization between language1 and multilingual tokenizer
        elif word_tokenization[0] == word_tokenization[2]:
            case_word_dict[f"{l1}_t==multi_t"].append(word)
            case_num_tokens_dict[f"{l1}_t==multi_t"].append([word, len(word_tokenization[0])])
        # Same tokenization between language2 and multilingual tokenizer
        elif word_tokenization[1] == word_tokenization[2]:
            case_word_dict[f"{l2}_t==multi_t"].append(word)
            case_num_tokens_dict[f"{l2}_t==multi_t"].append([word, len(word_tokenization[1])])
        # All different tokenization
        elif word_tokenization[0] != word_tokenization[1] and word_tokenization[0] != word_tokenization[2] and \
                word_tokenization[1] != word_tokenization[2]:
            case_word_dict["different_splits"].append(word)
            case_num_tokens_dict["different_splits"].append([word, len(word_tokenization[0]), len(word_tokenization[1]), len(word_tokenization[2])])
        # Same tokenization between language1 and langauge2, but different from Multi tokenizer
        elif word_tokenization[0] == word_tokenization[1]:
            case_word_dict[f"{l1}_t=={l2}_t"].append(word)
            case_num_tokens_dict[f"{l1}_t=={l2}_t"].append([word, len(word_tokenization[0])])
    
    plot_average_word_length(case_word_dict, algo, dir, l1, l2)
    plot_average_num_tokens(case_num_tokens_dict, algo, dir, l1, l2)
    plot_frequency_comparison(case_word_dict, algo, dir, word_frequencies, l1, l2)
    
def chi_square_test(ff_data, same_words_data, l1, l2, algo):
    """
    This function calculates the chi square test between the tokenization cases of False Friend words and the tokenization cases
    of words written the same in languages l1 and l2
    :param ff_data: the False Friends tokenization cases
    :param same_words_data: the same words across languages l1 and l2 tokenization cases
    :param l1: language 1
    :param l2: language 2
    :param algo: the name of the algorithm
    :return:
    """
    
    categories, ff_counts, same_words_counts = [], [], []
    for cat in ff_data.keys():
        categories.append(cat)
        ff_counts.append(ff_data[cat])
        same_words_counts.append(same_words_data[cat])
    
    # Create contingency table (2xN)
    contingency_table = np.array([ff_counts, same_words_counts])
    
    # Perform chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    # Print results
    print(f"Chi-Squared Test for {algo} on {l1}-{l2}")
    print(f"Categories: {categories}")
    print(f"False Friends counts: {ff_counts}")
    print(f"Same Words counts: {same_words_counts}")
    print(f"Chi-squared statistic: {chi2:.4f}")
    print(f"Degrees of freedom: {dof}")
    print(f"P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("✅ Statistically significant difference (p < 0.05)")
    else:
        print("❌ No statistically significant difference (p >= 0.05)")
    
    return chi2, dof, p_value, expected
    


def plot_tokenization_cases(num_tokens_diff, algo, l1, l2, word_types, dir):
    num_words = sum(num_tokens_diff.values())
    fig_save_path = f"{dir}/{word_types}_{l1}_{l2}_{algo}.png"
    title = f"Tokenization Cases:{l1}, {l2}\nAlgo:{algo}\nNum words: {num_words}"
    
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
    
def plot_average_word_length(case_word_dict, algo, dir, l1, l2):
    """
    This function plots the average word length and standard deviation of each tokenization category
    :param case_word_dict: a dictionary {tokenization_category: [list_of_words]}
    :param algo: the algo name
    :param dir: directory to save the figure
    :param l1: language 1
    :param l2: language 2
    :return:
    """
    categories = []
    means = []
    stds = []
    
    for category, words in case_word_dict.items():
        word_lengths = [len(w) for w in words]
        categories.append(category)
        means.append(np.mean(word_lengths) if word_lengths else 0)
        stds.append(np.std(word_lengths) if word_lengths else 0)
        
    
    fig_save_path = f"{dir}/graphs/avg_word_length_{l1}_{l2}_{algo}.png"
    title = f"Tokenization Cases - Average Word Length\nMean ± Std\n{l1}, {l2}\nAlgo:{algo}"
    
    plt.figure(figsize=(8, 6))
    x = np.arange(len(categories))
    plt.bar(x, means, yerr=stds, capsize=5, edgecolor='black')
    plt.xticks(x, categories, rotation=30, fontsize=12)
    plt.xlabel("Tokenization Case")
    plt.ylabel("Average Word Length")
    plt.title(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(fig_save_path)
    plt.show()
    
def plot_average_num_tokens(case_word_tokens_dict, algo, dir, l1, l2):
    """
    This function plots the mean and standard deviation of tokens for each tokenization case
    :param case_word_tokens_dict: a dictionary. Key is tokenization category, value is list of list. Each list looks like
    [word, len(word)]. For the different_splits category, the list looks like [word, len(l1_tokenization[0]), len(l2_tokenization[1]), len(l1_l2_tokenization[2])]
    :param algo: the algo name
    :param dir: directory to save the figure
    :param l1: language 1
    :param l2: language 2
    :return:
    """
    avg_token_length = []
    std_token_length = []
    categories = []
    
    for category, words in case_word_tokens_dict.items():
        if category != "different_splits":
            categories.append(category)
            num_tokens = [x[1] for x in words]
            if num_tokens:
                avg_token_length.append(np.mean(num_tokens))
                std_token_length.append(np.std(num_tokens))
            else:
                avg_token_length.append(0)
                std_token_length.append(0)
    
    # Different splits category
    l1_num_tokens, l2_num_tokens, l1_l2_num_tokens = [], [], []
    for word_tokenization in case_word_tokens_dict["different_splits"]:
        l1_num_tokens.append(word_tokenization[1])
        l2_num_tokens.append(word_tokenization[2])
        l1_l2_num_tokens.append(word_tokenization[3])
    
    categories.append("different_splits")
    if l1_num_tokens:
        avg_diff = [np.mean(l1_num_tokens), np.mean(l2_num_tokens), np.mean(l1_l2_num_tokens)]
        std_diff = [np.std(l1_num_tokens), np.std(l2_num_tokens), np.std(l1_l2_num_tokens)]
    else:
        avg_diff = [0, 0, 0]
        std_diff = [0, 0, 0]
    
    x = np.arange(len(categories))
    bar_width = 0.2
    plt.figure(figsize=(10, 6))
    
    # Plot regular bars with error bars
    for i, (mean, std) in enumerate(zip(avg_token_length, std_token_length)):
        plt.bar(x[i], mean, yerr=std, capsize=5, width=bar_width)
    
    # Plot grouped bars for "different_splits"
    group_labels = [l1, l2, f"{l1}_{l2}"]
    group_colors = ['lightblue', 'palegreen', 'khaki']
    group_offsets = [-bar_width, 0, bar_width]
    for i in range(3):
        plt.bar(x[-1] + group_offsets[i], avg_diff[i], yerr=std_diff[i],
                capsize=5, width=bar_width, color=group_colors[i], label=group_labels[i])
    
    fig_save_path = f"{dir}/graphs/avg_tokens_{l1}_{l2}_{algo}.png"
    plt.xticks(x, categories)
    plt.ylabel("Number of tokens")
    plt.title(f"Tokenization Cases - Average Tokens\nMean ± Std\n{l1}_{l2}_{algo}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_save_path)
    plt.show()
    
def plot_frequency_comparison(case_word_dict, algo, dir, word_frequencies, l1, l2):
    """
    This function plots a graph of the mean and standard deviation of the False Friends frequencies in the training corpus
    of the l1 and l2 languages
    :param case_word_dict: a dictionary {tokenization_category: [list_of_words]}
    :param algo: the algo name
    :param dir: directory to save the figure
    :param word_frequencies: word frequencies in the training corpus
    :param l1: language 1
    :param l2: language 2
    :return:
    """
    std1, std2 = [], []
    mean1, mean2 = [], []
    categories = []
    category_freq1 = dict()
    category_freq2 = dict()
    words_in_corpus = 0
    
    # Collect frequency data
    for category, words in case_word_dict.items():
        category_freq1[category] = []
        category_freq2[category] = []
        categories.append(category)
        for word in words:
            if word in word_frequencies[l1] and word in word_frequencies[l2]:
                category_freq1[category].append(word_frequencies[l1][word])
                category_freq2[category].append(word_frequencies[l2][word])
                words_in_corpus += 1
    
    # Compute mean and std
    for category in categories:
        freqs1 = category_freq1[category]
        freqs2 = category_freq2[category]
        mean1.append(np.mean(freqs1) if freqs1 else 0)
        std1.append(np.std(freqs1) if freqs1 else 0)
        mean2.append(np.mean(freqs2) if freqs2 else 0)
        std2.append(np.std(freqs2) if freqs2 else 0)
    
    # Plot
    fig_save_path = f"{dir}/graphs/frequencies_{l1}_{l2}_{algo}.png"
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    x = np.arange(len(categories))
    
    plt.bar(x - bar_width / 2, mean1, yerr=std1, capsize=5, width=bar_width, color='lightblue', label=f"{l1}")
    plt.bar(x + bar_width / 2, mean2, yerr=std2, capsize=5, width=bar_width, color='palegreen', label=f"{l2}")
    
    plt.xticks(x, categories, rotation=45)
    plt.ylabel("Frequency")
    plt.title(f"Tokenization Case Frequencies\nMean ± Std\n{l1}_{l2}_{algo}\nFalse Friends in Corpus: {words_in_corpus}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_save_path)
    plt.show()
    

def missing_ff_in_corpus(ff_data, word_frequencies, dir):
    """
    Creates a file of the missing False Friend words in the training corpus
    :param ff_data: the false friends data of a specific language
    :param word_frequencies: word frequencies of a specific language
    :param dir: the directory to save the .txt file
    :return:
    """
    
    with open(f"{dir}/missing_words.txt", 'w', encoding='utf-8') as f:
        for word in ff_data:
            if word not in word_frequencies.keys():
                f.write(f"{word}\n")

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
    for l, lang in languages_set.items():
        with open(f"./all_words_in_all_languages/{lang}/{lang}.txt", 'r', encoding='utf-8') as f:
            lines = f.readlines()[0].strip().lower().split(",")
        language_dict[l] = set(lines)
    
    for l in languages_set:
        if l != "en":
            ff_words = language_dict["en"] & language_dict[l]
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


language_set_map = {"en": "English", "fr": "French", "es": "Spanish", "de": "German", "se": "Swedish", "it": "Italian", "ro": "Romanian"}
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

same_words = get_same_words_across_languages(language_set_map)
# adding missing False Friends words to same words
for l in language_set_map:
    if l != "en":
        for ff in ff_data[l]:
            if ff not in same_words[l]:
                same_words[l].add(ff)
word_freq = get_word_frequencies_training_corpus("./training_data/words")

vocab_size = 3000
l1 = "en"
l1_tokenizers = dict()
for token_file in os.listdir(f"./experiments/{vocab_size}/{l1}"):
    file_name, file_type = token_file.split(".")
    algo = file_name.split("_", 1)[1]
    l1_tokenizers[algo] = load_tokenizer(f"./experiments/{vocab_size}/{l1}/{token_file}")

experiments_list = []
for e in os.listdir(f"./experiments/{vocab_size}"):
    if e != "en":
        experiments_list.append(e)
        
for l1_l2 in experiments_list:
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
        ff_tokenization_cases = analyze_tokenization([l1_tokenizer, l2_tokenizer, l1_l2_tokenizer], ff_data[l2], l1, l2, cur_algo)
        plot_tokenization_cases(ff_tokenization_cases, cur_algo, l1, l2, "ff", f"./analysis/{vocab_size}/{l1_l2}/graphs")
        write_tokenization_split([l1_tokenizer, l2_tokenizer, l1_l2_tokenizer], ff_data[l2], l1, l2, cur_algo,
                                 f"./analysis/{vocab_size}/{l1_l2}/tokenization/{cur_algo}.txt")
        same_words_tokenization_cases = analyze_tokenization([l1_tokenizer, l2_tokenizer, l1_l2_tokenizer], same_words[l2], l1, l2, cur_algo)
        # plot_tokenization_cases(same_words_tokenization_cases, cur_algo, l1, l2, "same_words", f"./analysis/{vocab_size}/{l1_l2}/graphs")

        intrinsic_analysis([l1_tokenizer, l2_tokenizer, l1_l2_tokenizer], ff_data[l2], word_freq, cur_algo, l1, l2, f"./analysis/{vocab_size}/{l1_l2}")
        chi_square_test(ff_tokenization_cases, same_words_tokenization_cases, l1, l2, cur_algo)
        print("##i######################################################################################################")
        
        







