# word to word id mapping
# source word lexicon and target word lexicon
SourceWLexicon = {}
TargetWLexicon = {}
# Reverse lookup table; ID to Word
RSourceWLexicon = {}
RTargetWLexicon = {}


def helper(file_path, lexicon, rlexicon, id):
    with open(file_path, 'r') as f:
        for line in f:
            words = line.split()
            for word in words:
                key = word.lower()
                if key not in lexicon:
                    lexicon[key] = id
                    rlexicon[id] = key
                    id += 1
    f.close()


def map_word_to_wordId(source_file, target_file):
    id = 0
    helper(source_file, SourceWLexicon, RSourceWLexicon, id)
    id = 0
    helper(target_file, TargetWLexicon, RTargetWLexicon, id)


SOURCE_PATH = 'data/target.txt'
TARGET_PATH = 'data/source.txt'
map_word_to_wordId(SOURCE_PATH, TARGET_PATH)
# print SourceWLexicon
# print TargetWLexicon
