import numpy as np
from scipy.stats import entropy
from util import SourceWLexicon, TargetWLexicon, SOURCE_PATH, TARGET_PATH


# V is vocabulary size
V = len(SourceWLexicon)
# Revised probablity matrix
RPMat = np.zeros((2, V, V))
# Initial uniform probability matrix
InMat = np.zeros((V, V))
InMat.fill(1/float(V))

# Expected count matrix; It's a sparse matrix
EcountMat = {}


def convert_word_to_wordId(source_line, target_line):
    """returns a list source word ids and target word
       ids.
    """
    source_words = source_line.split()
    target_words = target_line.split()
    source_id = [SourceWLexicon[word.lower()] for word in source_words]
    target_id = [TargetWLexicon[word.lower()] for word in target_words]
    return source_id, target_id


def build_freqTable(source_line, target_line):
    """takes source line and target line as parameter and
       returns frequency table for source and target words
    """
    source_w = {}
    target_w = {}
    source_id, target_id = convert_word_to_wordId(source_line, target_line)
    for id in source_id:
        source_w[id] = source_w.get(id, 0) + 1
    for id in target_id:
        target_w[id] = target_w.get(id, 0) + 1
    return source_w, target_w


def cal_expected_count(source_line, target_line):
    """expected count calculation using intitalised uniform
       probability. Storing expected count in EcountMat
    """
    global InMat, EcountMat
    source_w, target_w = build_freqTable(source_line, target_line)
    # check for unique words
    for s_id in source_w:
        for t_id in target_w:
            norm = sum([InMat[id][t_id] for id in source_w])
            p = InMat[s_id][t_id]/float(norm)
            exp_count = source_w[s_id] * target_w[t_id] * p
            key = (s_id, t_id)
            EcountMat[key] = exp_count


def fill_numDeno_RPMat():
    """Writes count(w, a) as numerator and
       column sum as denominator to RPMat matrix
    """
    global EcountMat, RPMat
    jth_col_sum = {}
    for key in EcountMat:
        i, j = key
        # adding numerator
        RPMat[0][i][j] = RPMat[0][i][j] + EcountMat[key]
        # adding denominator
        # denominator is sum over column j
        denom = 0
        if j not in jth_col_sum:
            for k in EcountMat:
                if k[1] == j:
                    denom = denom + EcountMat[k]
                    jth_col_sum[j] = denom
    for j in jth_col_sum:
        denom = jth_col_sum[j]
        RPMat[1][:, j:j+1] = RPMat[1][i][j] + denom


def clear_EcountMat():
    """clears EcountMat matrix
    """
    EcountMat.clear()


def revisedProbability(sf, tf):
    """calculates revised probability using
       the expected count values of pairs.
       Returns updated probability matrix
    """
    global RPMat
    for sl, tl in zip(sf, tf):
        cal_expected_count(sl, tl)
        fill_numDeno_RPMat()
        clear_EcountMat()
    return RPMat[0]/RPMat[1]


def calculate_entropy(mat):
    """calculates entropy for each column
       of the updated probability matrix.
    """
    entropy_colwise = []
    len_col = mat.shape[-1]
    for i in xrange(len_col):
        en = entropy(mat[:, i:i+1])
        entropy_colwise.append(en[0])
    return entropy_colwise


def main():
    """Driver to run the IBM model 1
    """
    sf = open(SOURCE_PATH, 'r')
    tf = open(TARGET_PATH, 'r')
    global InMat, RPMat
    c = 0
    while True:
        c += 1
        tmpMat = InMat[:, :]
        InMat = revisedProbability(sf, tf)
        print "Entropy for iteration %s: %s" % (c, calculate_entropy(InMat))
        # clear RPMat
        RPMat.fill(0.)
        sf.seek(0)
        tf.seek(0)
        if np.allclose(tmpMat, InMat):
            break
    sf.close()
    tf.close()
    print "========Final Prob Matrix======"
    print InMat
    print "Total Iteration: ", c


if __name__ == '__main__':
    main()
