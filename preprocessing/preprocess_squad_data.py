from textblob import TextBlob
import numpy as np

import argparse
import random
import json
import re

def writeSentenceLevelInputOutput(dataset, filename, vocab_file, isTrain=True):
    sentences_len = []
    paragraph_len = []
    num_paragraphs = 0
    num_ones = 0
    num_zeros = 0
    vocab = set([])

    f = open(filename, "w")
    for i in range(len(dataset)):
        num_paragraphs += len(dataset[i]['paragraphs'])
        for j in range(len(dataset[i]['paragraphs'])):
            context = dataset[i]['paragraphs'][j]['context'].encode('ascii','ignore')
            context = re.sub("\n", ". ", context)
            sentences = TextBlob(context.lower()).sentences
            sentences_end_index = []
            c_dist = 0
            if isTrain:
                paragraph_len.append(len(sentences))
            for s in range(len(sentences)):
                words = sentences[s].lower().words
                if isTrain:
                    sentences_len.append(len(words))
                    vocab.update(list(words))

                sent = sentences[s]
                c_dist += len(sent)
                sentences_end_index.append(c_dist)

            try:
                assert len(sentences) == len(sentences_end_index)
            except Exception as e:
                import pdb;pdb.set_trace()

            qas = dataset[i]['paragraphs'][j]['qas']
            for k in range(len(sentences_end_index)):
                answer_found = False
                for qa in qas:
                    if answer_found:
                        break
                    for ans in qa['answers']:
                        start = int(ans['answer_start'])
                        if ans['text'].lower() in sentences[k]:
                            if ((k > 0 and sentences_end_index[k-1] <= start and start < sentences_end_index[k]) \
                                or (k == 0 and start <= sentences_end_index[k])):
                                f.write(str(sentences[k]) + "\t" + "1\n")
                                num_ones += 1
                                answer_found = True
                                break
                if not answer_found:
                    f.write(str(sentences[k]) + "\t" + "0\n")
                    num_zeros += 1
            f.write("\n")
    print "Number of paragraphs: {}".format(num_paragraphs)
    print "Output (0) = {} (1) = {}".format(num_zeros, num_ones)
    if isTrain:
        with open(vocab_file,'w') as f:
            term_dict = {}
            term_dict["PAD"] = 0
            term_dict["UNK"] = 1
            for index, word in enumerate(list(vocab)):
                term_dict[word] = index+2
            json.dump(term_dict, f)

        print "Sentence Statistics:"
        print np.max(sentences_len), np.min(sentences_len), np.mean(sentences_len), np.median(sentences_len)
        print "Paragraph Statistics:"
        print np.max(paragraph_len)
        print "vocab size = {}".format(len(term_dict))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()

    c = json.load(open(args.config_file))
    squad_train_file = c["squad_train_file"]
    squad_dev_file = c["squad_dev_file"]
    sentlevel_train_file = c["sentlevel_train_file"]
    sentlevel_dev_file = c["sentlevel_dev_file"]
    sentlevel_test_file = c["sentlevel_test_file"]
    vocab_file = c["vocab_file"]

    random.seed(42)
    train_dataset = json.load(open(squad_train_file))
    dev_dataset = json.load(open(squad_dev_file))

    articles = dev_dataset['data']
    random.shuffle(articles)


    n = len(articles)
    test_index = int(0.5*n)
    new_dev_dataset = articles[:test_index]
    new_test_dataset = articles[test_index:]

    print "Total number of val+test articles: {}".format(n)
    print "val size: {}".format(len(new_dev_dataset))
    print "test size: {}".format(len(new_test_dataset))


    # train+val+test = 323 0 23
    # train+val+test = vocab size = 103399


    # train = 323 0 23.8613953601 22.0
    # train vocab size = 89976


    print "Generating sentence level data for Training"
    writeSentenceLevelInputOutput(train_dataset['data'], sentlevel_train_file, vocab_file)
    print "Generating sentence level data for Development"
    writeSentenceLevelInputOutput(new_dev_dataset, sentlevel_dev_file, vocab_file, isTrain=False)
    print "Generating sentence level data for Testing"
    writeSentenceLevelInputOutput(new_test_dataset, sentlevel_test_file, vocab_file, isTrain=False)


