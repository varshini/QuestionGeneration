from textblob import TextBlob
import argparse
import json

def writeHashedFile(filename, hashed_file, term_ids):
    with open(filename) as f:
        fw = open(hashed_file, "w")
        sent_count = 0
        for line in f:
            if line == "\n":
                #for i in range(sent_count, max_paragraph_length):
                #    fw.write(",".join([str(term_ids["PAD"])]*max_sentence_length) + "\t" + "0\n")
                fw.write(line)
                sent_count = 0
                continue
            print line
            sent = line.split("\t")[0]
            output = line.split("\t")[1]
            sent = TextBlob(sent).words
            hashed_sent = []
            for word in sent:
                if term_ids.get(word) == None:
                    hashed_sent.append(str(term_ids["UNK"]))
                else:
                    hashed_sent.append(str(term_ids[word]))
            hashed_sent += str(term_ids["PAD"])*(max_sentence_length - len(hashed_sent))
            hashed_sent = hashed_sent[:max_sentence_length]
            sent_count += 1
            if sent_count < max_paragraph_length:
                fw.write(",".join(hashed_sent) + "\t" + output)
        fw.close()

#PAD -> 0
#UNK -> 1
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")
    args = parser.parse_args()

    c = json.load(open(args.config_file))
    max_sentence_length = c["max_sentence_length"]
    max_paragraph_length = c["max_paragraph_length"]
    vocab_file = c["vocab_file"]
    sentlevel_train_file = c["sentlevel_train_file"]
    sentlevel_dev_file = c["sentlevel_dev_file"]
    sentlevel_test_file = c["sentlevel_test_file"]
    hashed_train_file = c["hashed_train_file"]
    hashed_dev_file = c["hashed_dev_file"]
    hashed_test_file = c["hashed_test_file"]

    term_ids = {}
    with open(vocab_file) as f:
        term_ids = json.load(f)

    writeHashedFile(sentlevel_train_file, hashed_train_file, term_ids)
    writeHashedFile(sentlevel_dev_file, hashed_dev_file, term_ids)
    writeHashedFile(sentlevel_test_file, hashed_test_file, term_ids)
