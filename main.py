# Promising false friends pairs:
# Note: From only starting the search for false friends language pairs, words that also sound the same but have a different meaning are also catagorized as false friends.
# English and Spanish have lots of false friends words that fall under that catagory, but, the words themselves seem to be spelled quite differently
# English-German https://en.wiktionary.org/wiki/Appendix:False_friends_between_English_and_German
# English-French https://en.wiktionary.org/wiki/Appendix:False_friends_between_English_and_French
import shutil
import sys
import os
import csv
import matplotlib.pyplot as plt
import random
from SaGe_main.src.sage_tokenizer import *
from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece, WordLevel
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer, WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


def analyze_tokenization(tokenizers_list, ff_data, l1, l2, algo, dir):
    """
    This function computes an analysis on how different tokenizers split False Friends words
    :param tokenizers_list: a list of tokenizers, [l1 tokenizer, l2 tokenizer, l1+l2 tokenizer]
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
    for i in range(len(ff_data)):
        ff = ff_data[i]["False Friend"]
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
    print(num_tokens_diff.items())
    x_axis = list(num_tokens_diff.keys())
    y_axis = [v for v in list(num_tokens_diff.values())]
    plt.bar(x_axis, y_axis)
    plt.xticks(rotation=30, fontsize=12)
    plt.xlabel("Tokenization Splits")
    plt.ylabel("Amount of Tokenization Case")
    plt.title(title, fontsize=15)
    plt.savefig(fig_save_path)
    plt.show()


def create_multi_text_file(path1, path2, file_name, num_rows, seed=42):
    """
    Creates a .txt file that combines two different text files by randomly sampling half of the lines
    from each input file using a specific random seed.

    :param path1: Path to file of first language
    :param path2: Path to file of second language
    :param file_name: Name of the combined output file
    :param num_rows: Total number of rows in the output file (half from each input)
    :param seed: Random seed for reproducibility
    """
    rows_from_each = num_rows // 2
    
    with open(path1, 'r', encoding='utf-8') as f1:
        lines1 = f1.readlines()
    with open(path2, 'r', encoding='utf-8') as f2:
        lines2 = f2.readlines()
    
    random.seed(seed)
    sampled1 = random.sample(lines1, rows_from_each)
    random.seed(seed + 1)
    sampled2 = random.sample(lines2, rows_from_each)
    
    with open(file_name, 'w', encoding='utf-8') as f_out:
        f_out.writelines(sampled1 + sampled2)

    
def get_SP_tokenizer(algo, vocab_size, corpus_file_path):
    """
    Get a trained tokenizer on a corpus
    :param algo: the type of tokenizer
    :param vocab_size: the size of the vocabulary
    :param corpus_file_path: a list that contains file paths of corpora
    :return: a trained tokenizer
    """
    
    if "BPE" in algo:
        tokenizer = Tokenizer(BPE(unk_token=unk_token))
        trainer = BpeTrainer(special_tokens=spl_tokens, vocab_size=vocab_size)
    elif "UNI" in algo:
        tokenizer = Tokenizer(Unigram())
        trainer = UnigramTrainer(unk_token=unk_token, special_tokens=spl_tokens, vocab_size=vocab_size)
    elif "WPC" in algo:
        tokenizer = Tokenizer(WordPiece(unk_token=unk_token))
        trainer = WordPieceTrainer(special_tokens=spl_tokens, vocab_size=vocab_size)
    else: #WLVL
        tokenizer = Tokenizer(WordLevel(unk_token=unk_token))
        trainer = WordLevelTrainer(special_tokens=spl_tokens, vocab_size=vocab_size)
    
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.train(corpus_file_path, trainer)
    return tokenizer
    
def get_sage_tokenizer(algo, schedule, initial_vocab_size, final_vocab_size, corpus_file_path, vocab_file_path, experiment_name):
    """
    
    :param algo:
    :param schedule:
    :param initial_vocab_size:
    :param final_vocab_size:
    :param corpus_file_path:
    :param vocab_file_path:
    :param experiment_name:
    :return:
    """
    vocab_builder_tokenizer = algo.split("_")[0]
    tokenizer = get_SP_tokenizer(vocab_builder_tokenizer, initial_vocab_size, corpus_file_path)
    vocab = sorted(list(tokenizer.get_vocab().keys()))
    hexed_vocab = add_single_bytes(hex_vocab(vocab))
    max_len = max([len(bytes.fromhex(str(v))) for v in hexed_vocab])
    
    with open(vocab_file_path, 'w', encoding='utf-8') as vocab_file:
        for hexed_v in hexed_vocab:
            vocab_file.write(f"{hexed_v}\n")
    
    trainer = SaGeVocabBuilder(full_vocab_schedule=schedule,
                               embeddings_schedule=schedule,
                               workers_number=4, max_len=max_len)
    
    trainer.build_vocab(experiment_name=experiment_name, corpus_filepath=corpus_file_path[0],
                        vocabulary_filepath=vocab_file_path)
    with open(f"./results/{experiment_name}/sage_vocabs/active_vocab_{final_vocab_size}.vocab", "r") as f:
        initial_vocab = [bytes.fromhex(line.strip()) for line in f]
    tokenizer = SaGeTokenizer(initial_vocabulary=initial_vocab)
    
    return tokenizer
        
        
def hex_vocab(vocab):
    """
    Translates the SaGE vocabulary to hexadecimal format
    :param vocab: list of vocabulary words generated by BPE or UNI or other tokenizers
    :return: list of hexadecimal vocabulary
    """
    hexed_vocab = []
    for v in vocab:
        hex_token = v.encode("utf-8").hex()
        hexed_vocab.append(hex_token)
    return hexed_vocab

def add_single_bytes(vocab):
    """
    SaGe requires all single bytes to be in the vocabulary. This function adds them in to the vocabulary in hexadecimal
    format, if needed
    :param vocab: list of hexadecimal vocabulary
    :return: updated vocabulary
    """
    for i in range(256):
        t = f"{i:02x}"
        if t not in vocab:
            vocab.append(t)
    return vocab

def create_experiments_dir(wd, l1, algorithms, experiments, vocab_size):
    for algo in algorithms:
        if "SAGE" in algo:
            if not os.path.exists(f"{wd}/results/{l1}_{algo}_{vocab_size}"):
                os.mkdir(f"{wd}/results/{l1}_{algo}_{vocab_size}")
                print(f"created directory {wd}/results/{l1}_{algo}_{vocab_size}")
    if not os.path.exists(f"{wd}/analysis"):
        os.mkdir(f"{wd}/analysis")
        print(f"created directory {wd}/analysis")
    if not os.path.exists(f"{wd}/analysis/{vocab_size}"):
        os.mkdir(f"{wd}/analysis/{vocab_size}")
        print(f"created directory {wd}/analysis/{vocab_size}")
    if not os.path.exists(f"{wd}/experiments"):
        os.mkdir(f"{wd}/experiments")
        print(f"created directory {wd}/experiments")
    if not os.path.exists(f"{wd}/experiments/{vocab_size}"):
        os.mkdir(f"{wd}/experiments/{vocab_size}")
        print(f"created directory {wd}/experiments/{vocab_size}")
    if not os.path.exists(f"{wd}/experiments/{vocab_size}/{l1}"):
        os.mkdir(f"{wd}/experiments/{vocab_size}/{l1}")
        print(f"created directory {wd}/experiments/{vocab_size}/{l1}")
    for experiment in experiments:
        l2 = experiment[0]
        for algo in algorithms:
            if not os.path.exists(f"{wd}/analysis/{vocab_size}/{l1}_{l2}"):
                os.mkdir(f"{wd}/analysis/{vocab_size}/{l1}_{l2}")
                print(f"created directory {wd}/analysis/{vocab_size}/{l1}_{l2}")
            if not os.path.exists(f"{wd}/analysis/{vocab_size}/{l1}_{l2}/graphs"):
                os.mkdir(f"{wd}/analysis/{vocab_size}/{l1}_{l2}/graphs")
                print(f"created directory {wd}/analysis/{vocab_size}/{l1}_{l2}/graphs")
            if not os.path.exists(f"{wd}/analysis/{vocab_size}/{l1}_{l2}/tokenization"):
                os.mkdir(f"{wd}/analysis/{vocab_size}/{l1}_{l2}/tokenization")
                print(f"created directory {wd}/analysis/{vocab_size}/{l1}_{l2}/tokenization")
            if not os.path.exists(f"{wd}/experiments/{vocab_size}/{l1}_{l2}"):
                os.mkdir(f"{wd}/experiments/{vocab_size}/{l1}_{l2}")
                print(f"created directory {wd}/experiments/{vocab_size}/{l1}_{l2}")
            if "SAGE" in algo:
                if not os.path.exists(f"{wd}/results/{l2}_{algo}_{vocab_size}"):
                    os.mkdir(f"{wd}/results/{l2}_{algo}_{vocab_size}")
                    print(f"created directory {wd}/results/{l2}_{algo}_{vocab_size}")
                if not os.path.exists(f"{wd}/results/{l1}_{l2}_{algo}_{vocab_size}"):
                    os.mkdir(f"{wd}/results/{l1}_{l2}_{algo}_{vocab_size}")
                    print(f"created directory {wd}/results/{l1}_{l2}_{algo}_{vocab_size}")

def get_experiments(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        experiments = f.readlines()
    for i in range(len(experiments)):
        l, train_data_path, ff_data_path = experiments[i].split(",")
        train_data_path = train_data_path.strip().replace("\\", "/")
        ff_data_path = ff_data_path.strip().replace("\\", "/")
        experiments[i] = (l, train_data_path, ff_data_path)
    return experiments

def save_tokenizer(tokenizer, dir):
    tokenizer.save(f"{dir}")

def train_l1_tokenizers(l1, algorithms, schedule, initial_vocab_size, final_vocab_size, corpus_file_path):
    
    en_tokenizers = dict()
    for algo in algorithms:
        if "SAGE" in algo:
            tokenizer = get_sage_tokenizer(algo, schedule, initial_vocab_size, final_vocab_size, corpus_file_path,
                                           f"./results/{l1}_{algo}_{vocab_size}/initial_vocab.vocab", f"{l1}_{algo}_{vocab_size}")
            shutil.copy(f"./results/{l1}_{algo}_{vocab_size}/sage_vocabs/active_vocab_{vocab_size}.vocab",
                        f"./experiments/{vocab_size}/{l1}/{l1}_{algo}.vocab")
        else:
            tokenizer = get_SP_tokenizer(algo, final_vocab_size, corpus_file_path)
            save_tokenizer(tokenizer, f"./experiments/{vocab_size}/{l1}/{l1}_{algo}.json")
        en_tokenizers[algo] = tokenizer
    return en_tokenizers
    
if __name__ == '__main__':
    
    num_rows = 300000
    vocab_size = 3000
    initial_sage_vocab_size = 8000
    schedule = [3000, 4000]
    unk_token = "<UNK>"  # token for unknown words
    spl_tokens = ["<UNK>", "<SEP>", "<MASK>", "<CLS>"]  # special tokens
    cwd = os.getcwd().replace("\\", "/")
    l1 = "en"
    experiments = get_experiments("./args_script.txt")
    algorithms, l1_file_path = sys.argv[1:]
    l1_file_path = l1_file_path.strip().replace("\\", "/")
    algorithms = algorithms.split(",")
    create_experiments_dir(cwd, l1, algorithms, experiments, vocab_size)
    # Training l1 tokenizers
    l1_tokenizers = train_l1_tokenizers(l1, algorithms, schedule, initial_sage_vocab_size, vocab_size, [l1_file_path])

    for experiment in experiments:
        l2, l2_file_path, ff_file_path = experiment
        multi_text_file = f"{cwd}/training_data/{l1}_{l2}/{l1}_{l2}_corpus.txt"
        print(f"created multi-text file {l1}_{l2}")
        # Creating multilingual training data
        create_multi_text_file(l1_file_path, l2_file_path, multi_text_file, num_rows)
        for algo in algorithms:
            print(f"starting experiment {l1}_{l2}_{algo}")
            if "SAGE" in algo:
                l2_tokenizer = get_sage_tokenizer(algo, schedule, initial_sage_vocab_size, vocab_size, [l2_file_path],
                                                  f"./results/{l2}_{algo}_{vocab_size}/initial_vocab.vocab", f"{l2}_{algo}_{vocab_size}")
                l1_l2_tokenizer = get_sage_tokenizer(algo, schedule, initial_sage_vocab_size, vocab_size, [multi_text_file],
                                                     f"./results/{l1}_{l2}_{algo}_{vocab_size}/initial_vocab.vocab", f"{l1}_{l2}_{algo}_{vocab_size}")

                shutil.copy(f"./results/{l2}_{algo}_{vocab_size}/sage_vocabs/active_vocab_{vocab_size}.vocab",
                            f"./experiments/{vocab_size}/{l1}_{l2}/{l2}_{algo}.vocab")
                shutil.copy(f"./results/{l1}_{l2}_{algo}_{vocab_size}/sage_vocabs/active_vocab_{vocab_size}.vocab",
                            f"./experiments/{vocab_size}/{l1}_{l2}/{l1}_{l2}_{algo}.vocab")
            else:
                l2_tokenizer = get_SP_tokenizer(algo, vocab_size, [l2_file_path])
                l1_l2_tokenizer = get_SP_tokenizer(algo, vocab_size, [multi_text_file])
                save_tokenizer(l2_tokenizer, f"{cwd}/experiments/{vocab_size}/{l1}_{l2}/{l2}_{algo}.json")
                save_tokenizer(l1_l2_tokenizer, f"{cwd}/experiments/{vocab_size}/{l1}_{l2}/{l1}_{l2}_{algo}.json")


            # Read False Friends data
            with open(ff_file_path, 'r', encoding='utf-8') as f:
                # list of dictionaries
                ff_data = list(csv.DictReader(f))

            analyze_tokenization([l1_tokenizers[algo], l2_tokenizer, l1_l2_tokenizer], ff_data, l1, l2,
                                 algo, f"{cwd}/analysis/{vocab_size}/{l1}_{l2}/graphs")
            # write_tokenization_split([l1_tokenizer, l2_tokenizer, l1_l2_tokenizer], ff_data, l1, l2, algo, f"{cwd}/analysis/{l1}_{l2}/tokenization/{l1}_{l2}_{algo}_ff.txt")
            print(f"finished experiment {l1}_{l2}_{algo}")
    