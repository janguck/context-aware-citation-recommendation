from aaerec.datasets import Bags
from aaerec.evaluation import Evaluation
from aaerec.aae import AAERecommender, DecodingRecommender
from aaerec.baselines import RandomBaseline, Countbased, MostPopular
from aaerec.svd import SVDRecommender
from gensim.models.keyedvectors import KeyedVectors
W2V_PATH = "./GoogleNews-vectors-negative300.bin.gz"
W2V_IS_BINARY = True
dataset = './dataset.tsv'
year = '2011'
outfile = './report.txt'
mincount = 10

DATASET = Bags.load_tabcomma_format(dataset, unique=False)
print("Dataset : ", DATASET)

EVAL = Evaluation(DATASET, year, logfile=outfile)
EVAL.setup(min_count=mincount, min_elements=2)
print("Loading pre-trained embedding", W2V_PATH)
VECTORS = KeyedVectors.load_word2vec_format(W2V_PATH, binary=W2V_IS_BINARY)

BASELINES = [
    Countbased(),
    SVDRecommender(1000, use_title=False),
]

ae_params = {
    'n_code': 50,
    'n_epochs': 100,
    'embedding': VECTORS,
    'batch_size': 100,
    'n_hidden': 100,
    'normalize_inputs': True,
}

RECOMMENDERS = [
    AAERecommender(use_title=False, adversarial=False, lr=0.0001,
                   **ae_params),
    AAERecommender(use_title=False, prior='gauss', gen_lr=0.0001,
                   reg_lr=0.0001, **ae_params),
]


TITLE_ENHANCED = [
    #SVDRecommender(1000, use_title=True),
    # DecodingRecommender(n_epochs=100, batch_size=100, optimizer='adam',
    #                     n_hidden=100, embedding=VECTORS,
    #                     lr=0.001, verbose=True),
    # AAERecommender(adversarial=False, use_title=True, lr=0.001,
    #                **ae_params),
    AAERecommender(adversarial=True, use_title=True,
                   prior='gauss', gen_lr=0.001, reg_lr=0.001,
                   **ae_params),
]

with open(outfile, 'a') as fh:
    print("~ Partial List", "~" * 42, file=fh)
#EVAL(BASELINES + RECOMMENDERS)
with open(outfile, 'a') as fh:
    print("~ Partial List + Titles", "~" * 42, file=fh)
EVAL(TITLE_ENHANCED)
