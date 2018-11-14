import tensorflow as tf
import numpy as np
from collections import Counter, namedtuple
import random
import zipfile
import os

Options = namedtuple('Options', 'data,eval,vocab_size,embed_size,skip_window,'+
    'learning_rate,batch_size,train_steps,num_sampled')

NAME = 'word2vec'

class Word2Vec:
    def __init__(self, sess, options):
        ''' Create W2V object with options. '''
        self._session = sess
        self.opts = options

    def load(self):
        ''' Loads in data for training. '''
        self.batch_gen = self.process_data(self.opts.vocab_size, self.opts.batch_size,
                                           self.opts.skip_window, self.opts.data)

    def build(self):
        ''' Build TF graph. '''
        # Define the placeholders for input and output
        self.center_words = tf.placeholder(tf.int32, shape=[self.opts.batch_size], name='center_words')
        self.target_words = tf.placeholder(tf.int32, shape=[self.opts.batch_size, 1], name='target_words')

        # Define weights. In word2vec, it's actually the weights that we care about
        self.embed_matrix = tf.Variable(tf.random_uniform([self.opts.vocab_size, self.opts.embed_size], -1.0, 1.0), 
                                name='embed_matrix')

        # Define the inference
        embed = tf.nn.embedding_lookup(self.embed_matrix, self.center_words, name='embed')

        # Construct variables for NCE loss
        nce_weight = tf.Variable(tf.truncated_normal([self.opts.vocab_size, self.opts.embed_size],
                                                    stddev=1.0 / (self.opts.embed_size ** 0.5)), 
                                                    name='nce_weight')
        nce_bias = tf.Variable(tf.zeros([self.opts.vocab_size]), name='nce_bias')

        # Define loss function to be NCE loss function
        self.loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight, 
                                                  biases=nce_bias, 
                                                  labels=self.target_words, 
                                                  inputs=embed, 
                                                  num_sampled=self.opts.num_sampled, 
                                                  num_classes=self.opts.vocab_size), name='loss')

        # Define optimizer
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.optimizer = tf.train.GradientDescentOptimizer(self.opts.learning_rate).minimize(self.loss, global_step=self.global_step)

    def build_eval_graph(self):
        """Build the eval graph."""
        # Eval graph

        # Each analogy task is to predict the 4th word (d) given three
        # words: a, b, c.  E.g., a=italy, b=rome, c=france, we should
        # predict d=paris.

        # The eval feeds three vectors of word ids for a, b, c, each of
        # which is of size N, where N is the number of analogies we want to
        # evaluate in one batch.
        analogy_a = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_b = tf.placeholder(dtype=tf.int32)  # [N]
        analogy_c = tf.placeholder(dtype=tf.int32)  # [N]

        # Normalized word embeddings of shape [vocab_size, emb_dim].
        nemb = tf.nn.l2_normalize(self.embed_matrix, 1)

        # Each row of a_emb, b_emb, c_emb is a word's embedding vector.
        # They all have the shape [N, emb_dim]
        a_emb = tf.gather(nemb, analogy_a)  # a's embs
        b_emb = tf.gather(nemb, analogy_b)  # b's embs
        c_emb = tf.gather(nemb, analogy_c)  # c's embs

        # We expect that d's embedding vectors on the unit hyper-sphere is
        # near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
        target = c_emb + (b_emb - a_emb)

        # Compute cosine distance between each pair of target and vocab.
        # dist has shape [N, vocab_size].
        dist = tf.matmul(target, nemb, transpose_b=True)

        # For each question (row in dist), find the top 4 words.
        _, pred_idx = tf.nn.top_k(dist, 4)

        # Nodes for computing neighbors for a given word according to
        # their cosine distance.
        nearby_word = tf.placeholder(dtype=tf.int32)  # word id
        nearby_emb = tf.gather(nemb, nearby_word)
        nearby_dist = tf.matmul(nearby_emb, nemb, transpose_b=True)
        nearby_val, nearby_idx = tf.nn.top_k(nearby_dist,
                                             min(1000, self.opts.vocab_size))

        # Nodes in the construct graph which are used by training and
        # evaluation to run/feed/fetch.
        self._analogy_a = analogy_a
        self._analogy_b = analogy_b
        self._analogy_c = analogy_c
        self._analogy_pred_idx = pred_idx
        self._nearby_word = nearby_word
        self._nearby_val = nearby_val
        self._nearby_idx = nearby_idx

    def train(self):
        ''' Train the model. '''
        modelName = 'word2vec' + ','.join([f'{k}={v}' for k, v in self.opts._asdict().items() if k not in ['eval', 'data']])
        ckptPath = f'out/{NAME}/checkpoints/{modelName}.ckpt'
        sess = self._session
        sess.run(tf.global_variables_initializer())

        ## Restore checkpoint and write graph
        saver = tf.train.Saver()
        '''
        ckpt = tf.train.get_checkpoint_state(f'out/{NAME}/checkpoints/')
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path) 
            print("Restoring from checkpoint")
            saver.restore(sess, ckptPath)  
            '''
        writer = tf.summary.FileWriter(f'out/{NAME}/graphs/{modelName}', sess.graph)
        ##

        total_loss = 0.0 # we use this to calculate late average loss in the last SKIP_STEP steps
        #for index in range(NUM_TRAIN_STEPS):
        index = self.global_step.eval()
        print(f"Start index = {index}")
        SKIP_STEP = 1000
        while index < self.opts.train_steps:
            centers, targets = next(self.batch_gen)
            loss_batch, _ = sess.run([self.loss, self.optimizer], 
                                    feed_dict={
                                        self.center_words: centers,
                                        self.target_words: targets
                                    })
            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                saver.save(sess, ckptPath)
                total_loss = 0.0
            index += 1
        centers, targets = next(self.batch_gen)
        feed_dict = {
            self.center_words: centers,
            self.target_words: targets
        }
        print("Final loss = {:.2f}".format(sess.run(self.loss, feed_dict=feed_dict)))
        saver.save(sess, ckptPath)
        writer.close()

    def eval(self):
        ''' Evaluate model on loaded analogies. '''
        self.read_analogies()

        # How many questions we get right at
        total = self._analogy_questions.shape[0]
        correct = 0
        start = 0
        while start < total:
          limit = start + 2500
          sub = self._analogy_questions[start:limit, :]
          idx = self._predict(sub)
          start = limit
          for question in range(sub.shape[0]):
            for j in range(4):
              if start < 10000:
                  print('In: {}'.format(list(map(lambda x: self.rev[x], sub[question]))))
                  print('Out: {}'.format(list(map(lambda x: self.rev[x], idx[question]))))
              if idx[question, j] == sub[question, 3]:
                # Bingo! We predicted correctly. E.g., [italy, rome, france, paris].
                correct += 1
                break
              elif idx[question, j] in sub[question, :3]:
                # We need to skip words already in the question.
                continue
              else:
                # The correct label is not the precision@1
                break
        print()
        print("Eval %4d/%d accuracy = %4.1f%%" % (correct, total,
                                                  correct * 100.0 / total))        

    def _predict(self, analogy):
        """Predict the top 4 answers for analogy questions."""
        idx, = self._session.run([self._analogy_pred_idx], {
            self._analogy_a: analogy[:, 0],
            self._analogy_b: analogy[:, 1],
            self._analogy_c: analogy[:, 2]
        })
        return idx


    def read_analogies(self):
        '''Reads through the analogy question file.
        Returns:
          questions: a [n, 4] numpy array containing the analogy question's
                     word ids.
          questions_skipped: questions skipped due to unknown words.
        '''
        questions = []
        questions_skipped = 0
        with open(self.opts.eval, "r") as analogy_f:
            for line in analogy_f:
                if line.startswith(":"):  # Skip comments.
                    continue
                words = line.strip().lower().split(" ")
                ids = [self.dictionary.get(w.strip().lower()) for w in words]
                if None in ids or len(ids) != 4:
                    questions_skipped += 1
                else:
                  questions.append(np.array(ids))
            print("Eval analogy file: ", self.opts.eval)
            print("Questions: ", len(questions))
            print("Skipped: ", questions_skipped)
            self._analogy_questions = np.array(questions, dtype=np.int32)  

    ## Process Data ##
    def process_data(self, vocab_size, batch_size, skip_window, fn):
        ''' Returns batch generator for wikitext data. '''
        words = self.load_data(fn)
        self.dictionary, self.rev = self.build_vocab(words, vocab_size)
        index_words = self.convert_words_to_index(words, self.dictionary)
        del words
        single_gen = self.generate_sample(index_words, skip_window)
        return self.get_batch(single_gen, batch_size)


    @staticmethod
    def load_data(fn):
        ''' Reads the wikitext file into a list of tokens. '''
        if fn.endswith('.zip'):
            with zipfile.ZipFile(fn) as f:
                words = tf.compat.as_str(f.read(f.namelist()[0])).split() 
        else:
            with open(fn) as f:
                words = tf.compat.as_str(f.read()).split() 
        return words

    @staticmethod
    def build_vocab(words, vocab_size):
        ''' Build vocabulary of vocab_size most frequent words. '''
        dictionary = dict()
        count = [('UNK', -1)]
        count.extend(Counter(words).most_common(vocab_size - 1))
        index = 0
        for word, _ in count:
            dictionary[word] = index
            index += 1
        rev = dict(zip(dictionary.values(), dictionary.keys()))
        return dictionary, rev

    @staticmethod
    def convert_words_to_index(words, dictionary):
        ''' Converts list of words into indices in corresponding dict. '''
        return [dictionary[word] if word in dictionary else 0 for word in words]

    @staticmethod
    def generate_sample(index_words, context_window_size):
        """ Form training pairs according to the skip-gram model. """
        for index, center in enumerate(index_words):
            # Random context. Acts to subsample more distant words.
            context = random.randint(1, context_window_size)
            # Get pairs within context
            for target in index_words[max(0, index - context): index]:
                yield center, target
            for target in index_words[index + 1: index + context + 1]:
                yield center, target

    @staticmethod
    def get_batch(iterator, batch_size):
        """ Group a numerical stream into batches and yield them as Numpy arrays. """
        while True:
            center_batch = np.zeros(batch_size, dtype=np.int32)
            target_batch = np.zeros([batch_size, 1])
            for index in range(batch_size):
                center_batch[index], target_batch[index] = next(iterator)
            yield center_batch, target_batch

#def main():
opts = Options(
    data='data/wiki_text/text8.zip',
    eval='data/analogies.txt',
    vocab_size=7000,
    embed_size=200,
    skip_window=5,
    learning_rate=0.2,
    batch_size=16,
    train_steps=200000,
    num_sampled=100
)

with tf.Session() as sess:
    print("START")
    w2v = Word2Vec(sess, opts)
    w2v.load()
    print("Loaded")
    w2v.build()
    w2v.build_eval_graph()
    print("Built")
    w2v.train()
    print("Trained")
    w2v.eval()
    print("Eval")

#if __name__ == '__main__':
#    main()