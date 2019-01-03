from keras import backend as K
from keras.layers.recurrent import Recurrent
from keras.engine import InputSpec
from keras.models import Model
from keras import regularizers, constraints, initializers, activations
from keras.layers import *

class NCN_GRU(Recurrent):

    def __init__(self, units, encoder=None,
                 activation='tanh',
                 return_probabilities=False,
                 name='AttentionDecoder',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        self.units = units
        self.encoder = encoder
        self.return_probabilities = return_probabilities
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(NCN_GRU, self).__init__(**kwargs)
        self.name = name
        self.return_sequences = True

    def build(self, input_shape):

        self.batch_size, self.encoder_num, self.encoder_dim = input_shape[0]
        self.batch_size, self.timesteps, self.input_dim = input_shape[1]

        if self.stateful:
            super(NCN_GRU, self).reset_states()

        self.states = [None, None]

        self.V_a = self.add_weight(shape=(self.units,),
                                   name='V_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.W_a = self.add_weight(shape=(self.units, self.units),
                                   name='W_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_a = self.add_weight(shape=(self.units, self.units),
                                   name='U_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.W_r = self.add_weight(shape=(self.units, self.units),
                                   name='W_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.V_r = self.add_weight(shape=(self.input_dim, self.units),
                                   name='V_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_r = self.add_weight(shape=(self.units, self.units),
                                   name='U_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_r = self.add_weight(shape=(self.units, ),
                                   name='b_r',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        self.W_z = self.add_weight(shape=(self.input_dim, self.units),
                                   name='W_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.V_z = self.add_weight(shape=(self.input_dim, self.units),
                                   name='V_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_z = self.add_weight(shape=(self.units, self.units),
                                   name='U_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_z = self.add_weight(shape=(self.units, ),
                                   name='b_z',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        self.W_o = self.add_weight(shape=(self.input_dim, self.units),
                                   name='W_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.V_o = self.add_weight(shape=(self.input_dim, self.units),
                                   name='V_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_o = self.add_weight(shape=(self.units, self.units),
                                   name='U_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_p = self.add_weight(shape=(self.units, ),
                                   name='b_p',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        self.input_spec = [
            InputSpec(shape=(self.batch_size, self.encoder_num, self.encoder_dim)),
            InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))
        ]
        self.built = True

    def call(self, x):

        cnn_encoder, x_seq = x

        self.encoder_dot = K.dot(cnn_encoder, self.U_a)
        self.x_seq = x[0]

        return super(NCN_GRU, self).call(x[0])


    def step(self, x, states):
        htm, stm = states

        W_a_previous_dot = K.dot(K.expand_dims(stm, axis=1), self.W_a)

        et = K.dot(activations.tanh(W_a_previous_dot + self.encoder_dot),
                   K.expand_dims(self.V_a))
        et = Lambda(lambda x: K.softmax(x))(et)
        context = Lambda(lambda x: dot([x[0], x[1]], axes=[1, 1]))([et, self.encoder_dot])
        context = Flatten()(context)
        ri = activations.sigmoid(
            K.dot(x, self.W_r)
            + K.dot(stm, self.U_r)
            + K.dot(context, self.V_r)
            + self.b_r)
        zi = activations.sigmoid(
            K.dot(x, self.W_z)
            + K.dot(stm, self.U_z)
            + K.dot(context, self.V_z)
            + self.b_z)
        st_ocean = activations.tanh(
            K.dot(x, self.W_o)
            + (ri * K.dot(stm, self.U_o))
            + K.dot(context, self.V_o)
            + self.b_p)
        st = (1-zi)*st_ocean + zi * stm

        return htm, [htm, st]
    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_probabilities:
            return (None, self.timesteps, self.timesteps)
        else:
            return (None, self.timesteps, self.units)

if __name__ == '__main__':

    ### ------------------Encoder------------------ ###
    ENCODER_VOCUBALARY_SIZE = 20002
    EMBEDDING_SIZE = 64
    ENCODER_SEQUENCE_LENGTH = 10

    encoder_citation_context = Input(shape=(ENCODER_SEQUENCE_LENGTH,), dtype='int32', name='citation_context')
    word_embedding_layer = Embedding(input_dim=EMBEDDING_SIZE, output_dim=EMBEDDING_SIZE)
    encoder_citation_context_layer = word_embedding_layer(encoder_citation_context)

    encoder_citation_context_filter_sizes = [4, 4, 5]
    encoder_citation_context_num_filters = 64
    encoder_citation_context_layer = Reshape((ENCODER_SEQUENCE_LENGTH, EMBEDDING_SIZE, 1))(encoder_citation_context_layer)

    encoder_citation_context_conv_0 = Conv2D(encoder_citation_context_num_filters, kernel_size=(encoder_citation_context_filter_sizes[0], EMBEDDING_SIZE),
                            padding='valid', kernel_initializer='normal', activation='relu')(encoder_citation_context_layer)
    encoder_citation_context_conv_1 = Conv2D(encoder_citation_context_num_filters, kernel_size=(encoder_citation_context_filter_sizes[1], EMBEDDING_SIZE),
                            padding='valid', kernel_initializer='normal', activation='relu')(encoder_citation_context_layer)
    encoder_citation_context_conv_2 = Conv2D(encoder_citation_context_num_filters, kernel_size=(encoder_citation_context_filter_sizes[2], EMBEDDING_SIZE),
                            padding='valid', kernel_initializer='normal', activation='relu')(encoder_citation_context_layer)

    encoder_citation_context_maxpool_0 = MaxPool2D(pool_size=(ENCODER_SEQUENCE_LENGTH - encoder_citation_context_filter_sizes[0] + 1, 1), strides=(1, 1),
                                  padding='valid')(encoder_citation_context_conv_0)
    encoder_citation_context_maxpool_1 = MaxPool2D(pool_size=(ENCODER_SEQUENCE_LENGTH - encoder_citation_context_filter_sizes[1] + 1, 1), strides=(1, 1),
                                  padding='valid')(encoder_citation_context_conv_1)
    encoder_citation_context_maxpool_2 = MaxPool2D(pool_size=(ENCODER_SEQUENCE_LENGTH - encoder_citation_context_filter_sizes[2] + 1, 1), strides=(1, 1),
                                  padding='valid')(encoder_citation_context_conv_2)

    ### ------------------Decoder------------------ ###

    DECODER_AUTHOR_SIZE = 20001
    DECODER_AUTHOR_LENGTH = 5

    decoder_citing_author_inputs = Input(shape=(DECODER_AUTHOR_LENGTH,), dtype='int32', name='citing_author')
    decoder_cited_author_inputs = Input(shape=(DECODER_AUTHOR_LENGTH,), dtype='int32', name='cited_author')

    author_embedding_layer = Embedding(input_dim=EMBEDDING_SIZE, output_dim=EMBEDDING_SIZE)

    decoder_filter_sizes = [1, 2]
    decoder_num_filters = 64

    decoder_citing_author_layer = author_embedding_layer(decoder_citing_author_inputs)
    decoder_cited_author_layer = author_embedding_layer(decoder_cited_author_inputs)

    decoder_citing_author_layer = Reshape((DECODER_AUTHOR_LENGTH, EMBEDDING_SIZE, 1))(decoder_citing_author_layer)
    decoder_cited_author_layer = Reshape((DECODER_AUTHOR_LENGTH, EMBEDDING_SIZE, 1))(decoder_cited_author_layer)




    decoder_citing_author_conv_0 = Conv2D(decoder_num_filters, kernel_size=(decoder_filter_sizes[0], EMBEDDING_SIZE),
                                    padding='valid', kernel_initializer='normal', activation='relu')(decoder_citing_author_layer)
    decoder_citing_author_conv_1 = Conv2D(decoder_num_filters, kernel_size=(decoder_filter_sizes[1], EMBEDDING_SIZE),
                                    padding='valid', kernel_initializer='normal', activation='relu')(decoder_citing_author_layer)

    decoder_cited_author_conv_0 = Conv2D(decoder_num_filters, kernel_size=(decoder_filter_sizes[0], EMBEDDING_SIZE),
                                          padding='valid', kernel_initializer='normal', activation='relu')(decoder_cited_author_layer)
    decoder_cited_author_conv_1 = Conv2D(decoder_num_filters, kernel_size=(decoder_filter_sizes[1], EMBEDDING_SIZE),
                                          padding='valid', kernel_initializer='normal', activation='relu')(decoder_cited_author_layer)

    decoder_citing_author_maxpool_0 = MaxPool2D(pool_size=(DECODER_AUTHOR_LENGTH - decoder_filter_sizes[0] + 1, 1), strides=(1, 1),
                                  padding='valid')(decoder_citing_author_conv_0)
    decoder_citing_author_maxpool_1 = MaxPool2D(pool_size=(DECODER_AUTHOR_LENGTH - decoder_filter_sizes[1] + 1, 1), strides=(1, 1),
                                  padding='valid')(decoder_citing_author_conv_1)

    decoder_cited_author_maxpool_0 = MaxPool2D(pool_size=(DECODER_AUTHOR_LENGTH - decoder_filter_sizes[0] + 1, 1),
                                                strides=(1, 1),
                                                padding='valid')(decoder_cited_author_conv_0)
    decoder_cited_author_maxpool_1 = MaxPool2D(pool_size=(DECODER_AUTHOR_LENGTH - decoder_filter_sizes[1] + 1, 1),
                                                strides=(1, 1),
                                                padding='valid')(decoder_cited_author_conv_1)




    ### ------------------Merge------------------ ###
    concatenated = Concatenate(axis=1)([encoder_citation_context_maxpool_0, encoder_citation_context_maxpool_1, encoder_citation_context_maxpool_2,
                                        decoder_citing_author_maxpool_0, decoder_citing_author_maxpool_1,
                                        decoder_cited_author_maxpool_0, decoder_cited_author_maxpool_1])
    concatenated = Lambda(lambda  x: K.squeeze(x, 2))(concatenated)
    ### -------------------------------------------------------------------------------------- ###
    DECODER_TITLE_LEN = 17
    RECOMMENDATION_NUM = 2000
    title_decoder_inputs = Input(shape=(DECODER_TITLE_LEN,), dtype='int32', name="title_decoder_inputs")
    title_decoder_layer = word_embedding_layer(title_decoder_inputs)
    output = NCN_GRU(EMBEDDING_SIZE)([concatenated, title_decoder_layer])
    output_softmax = Dense(RECOMMENDATION_NUM, activation="softmax")(output)
    model = Model(inputs=[encoder_citation_context, decoder_citing_author_inputs, decoder_cited_author_inputs,
                          title_decoder_inputs], outputs=output_softmax)
    model.summary()
