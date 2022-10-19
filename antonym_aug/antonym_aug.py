import random
from nlp_aug import *
from nltk.corpus import wordnet

from nlp_aug import stop_words

def antonym_replacement(words, n):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words]))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        antonyms = get_antonym(random_word)
        if len(antonyms) >= 1:
            antonym = random.choice(list(antonyms))
            new_words = ['not ' + antonym if word == random_word else word for word in new_words]
            # print("replaced", random_word, "with not ", antonym)
            num_replaced += 1
        if num_replaced >= n: #only replace up to n words
            break

    #this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words

def get_antonym(word):
    antonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            for ant in l.antonyms():
                antonym = ant.name().replace('_', ' ').replace(' ', ' ').lower()
                antonym = "".join([char for char in antonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
                antonyms.add(antonym)
    if word in antonyms:
        antonyms.remove(word)
    return list(antonyms)

def AR(sentence, alpha=0.3, num_aug=1):
    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word is not '']
    num_words = len(words)

    augmented_sentences = []
    n = max(1, int(alpha*num_words))

    for _ in range(num_aug):
        a_words = antonym_replacement(words, n)
        augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    random.shuffle(augmented_sentences)

    augmented_sentences.append(sentence)

    return augmented_sentences

def eda_5(sentence, alpha_sr=0.3, alpha_ri=0.2, alpha_rs=0.1, p_rd=0.15, alpha_ar=0.3, num_aug=9):
	
    sentence = get_only_chars(sentence)
    words = sentence.split(' ')
    words = [word for word in words if word is not '']
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug/5)+1
    n_sr = max(1, int(alpha_sr*num_words))
    n_ri = max(1, int(alpha_ri*num_words))
    n_rs = max(1, int(alpha_rs*num_words))
    n_ar = max(1, int(alpha_ar*num_words))

	#sr
    for _ in range(num_new_per_technique):
        a_words = synonym_replacement(words, n_sr)
        augmented_sentences.append(' '.join(a_words))

	#ri
    for _ in range(num_new_per_technique):
        a_words = random_addition(words, n_ri)
        augmented_sentences.append(' '.join(a_words))

	#rs
    for _ in range(num_new_per_technique):
        a_words = random_swap(words, n_rs)
        augmented_sentences.append(' '.join(a_words))

	#rd
    for _ in range(num_new_per_technique):
        a_words = random_deletion(words, p_rd)
        augmented_sentences.append(' '.join(a_words))

    # ar
    for _ in range(num_new_per_technique):
        a_words = antonym_replacement(words, n_ar)
        augmented_sentences.append(' '.join(a_words))

    augmented_sentences = [get_only_chars(sentence) for sentence in augmented_sentences]
    shuffle(augmented_sentences)

	#trim so that we have the desired number of augmented sentences
    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

	#append the original sentence
    augmented_sentences.append(sentence)

    return augmented_sentences