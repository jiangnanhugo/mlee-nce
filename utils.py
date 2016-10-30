import sys, logging
from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger('food2vec')

def loadWord2Vec(filename):
    try:
        logger.info("Trying to load food2vec model from file: {0}".format(filename))
        food2vec = Word2Vec.load_word2vec_format(filename, binary=False)
        logger.info("Food2vec model has been loaded from file: {0}".format(filename))
        return food2vec
    except IOError as e:
        logger.error("Cannot load food2vec model from file: {0}: IOError: {1}".format(filename, e.strerror))
        sys.exit(e.errno)

def saveWord2Vec(filename, model):
    try:
        model.save_word2vec_format(filename, binary=False)
        logger.info('Model has been saved!')
    except IOError as e:
        logger.error("Cannot save food2vec model: IOError: {0}".format(e.strerror))