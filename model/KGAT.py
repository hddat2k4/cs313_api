'''
Created on Dec 18, 2018
Tensorflow Implementation of Knowledge Graph Attention Network (KGAT) model in:
Wang Xiang et al. KGAT: Knowledge Graph Attention Network for Recommendation. In KDD 2019.
@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import tensorflow as tf
# TF1_COMPAT: Import v1 compatibility module
import tensorflow.compat.v1 as tf_v1
# tf_v1.disable_eager_execution() # Optional: Already disabled in Main.py

import os
import numpy as np
import scipy.sparse as sp
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class KGAT(object):
    def __init__(self, data_config, pretrain_data, args):
        self._parse_args(data_config, pretrain_data, args)
        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        self._build_inputs()

        """
        *********************************************************
        Create Model Parameters for CF & KGE parts.
        """
        self.weights = self._build_weights()

        """
        *********************************************************
        Compute Graph-based Representations of All Users & Items & KG Entities via Message-Passing Mechanism of Graph Neural Networks.
        Different Convolutional Layers:
            1. bi: defined in 'KGAT: Knowledge Graph Attention Network for Recommendation', KDD2019;
            2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
            3. graphsage: defined in 'Inductive Representation Learning on Large Graphs', NeurIPS2017.
        """
        self._build_model_phase_I()
        """
        Optimize Recommendation (CF) Part via BPR Loss.
        """
        self._build_loss_phase_I()

        """
        *********************************************************
        Compute Knowledge Graph Embeddings via TransR.
        """
        self._build_model_phase_II()
        """
        Optimize KGE Part via BPR Loss.
        """
        self._build_loss_phase_II()

        self._statistics_params()

    def _parse_args(self, data_config, pretrain_data, args):
        # argument settings
        self.model_type = 'kgat' # Keep internal model type string

        self.pretrain_data = pretrain_data

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_entities = data_config['n_entities']
        self.n_relations = data_config['n_relations']

        self.n_fold = 100 # Define n_fold based on args or default

        # initialize the attentive matrix A for phase I.
        # Ensure A_in is a scipy sparse matrix or compatible type
        self.A_in = data_config.get('A_in', None) # Use .get for safety
        if self.A_in is None:
            print("Warning: A_in not found in data_config. Model might fail.")
            # Handle initialization appropriately if A_in is missing

        self.all_h_list = data_config.get('all_h_list', [])
        self.all_r_list = data_config.get('all_r_list', [])
        self.all_t_list = data_config.get('all_t_list', [])
        self.all_v_list = data_config.get('all_v_list', []) # For attention scores

        self.adj_uni_type = args.adj_uni_type

        self.lr = args.lr

        # settings for CF part.
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size

        # settings for KG part.
        self.kge_dim = args.kge_size
        self.batch_size_kg = args.batch_size_kg

        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)

        self.alg_type = args.alg_type
        # Keep model type string logic if used elsewhere, but self.model_type = 'kgat' is simpler
        # self.model_type += '_%s_%s_%s_l%d' % (args.adj_type, args.adj_uni_type, args.alg_type, self.n_layers)

        self.regs = eval(args.regs)
        self.verbose = args.verbose

        # Check necessary args are present
        if not hasattr(args, 'adj_type'): args.adj_type = 'default_adj_type' # Add defaults if missing
        if not hasattr(args, 'adj_uni_type'): args.adj_uni_type = 'default_uni_type'


    def _build_inputs(self):
        # placeholder definition
        # TF1_COMPAT: Use v1 placeholder
        self.users = tf_v1.placeholder(tf.int32, shape=(None,), name='users')
        self.pos_items = tf_v1.placeholder(tf.int32, shape=(None,), name='pos_items')
        self.neg_items = tf_v1.placeholder(tf.int32, shape=(None,), name='neg_items')

        # for knowledge graph modeling (TransD / Attentive KG)
        # TF1_COMPAT: Use v1 placeholder
        self.A_values = tf_v1.placeholder(tf.float32, shape=[len(self.all_v_list)], name='A_values')

        # TF1_COMPAT: Use v1 placeholder
        self.h = tf_v1.placeholder(tf.int32, shape=[None], name='h')
        self.r = tf_v1.placeholder(tf.int32, shape=[None], name='r')
        self.pos_t = tf_v1.placeholder(tf.int32, shape=[None], name='pos_t')
        self.neg_t = tf_v1.placeholder(tf.int32, shape=[None], name='neg_t')

        # dropout: node dropout (adopted on the ego-networks);
        # message dropout (adopted on the convolution operations).
        # TF1_COMPAT: Use v1 placeholder
        self.node_dropout = tf_v1.placeholder(tf.float32, shape=[None], name='node_dropout')
        self.mess_dropout = tf_v1.placeholder(tf.float32, shape=[None], name='mess_dropout')

    def _build_weights(self):
        all_weights = dict()

        # TF1_COMPAT: Use v1 initializers
        # initializer = tf.contrib.layers.xavier_initializer() # Deprecated
        initializer = tf_v1.initializers.glorot_uniform() # Xavier uniform initializer

        if self.pretrain_data is None:
            # TF1_COMPAT: Use v1 Variable
            all_weights['user_embed'] = tf_v1.Variable(initializer([self.n_users, self.emb_dim]), name='user_embed')
            all_weights['entity_embed'] = tf_v1.Variable(initializer([self.n_entities, self.emb_dim]), name='entity_embed')
            print('using xavier initialization')
        else:
            # TF1_COMPAT: Use v1 Variable
            all_weights['user_embed'] = tf_v1.Variable(initial_value=self.pretrain_data['user_embed'], trainable=True,
                                                    name='user_embed', dtype=tf.float32)

            # Ensure item_embed shape matches expected part of entity_embed
            item_embed = self.pretrain_data['item_embed']
            num_items = item_embed.shape[0]
            num_other_entities = self.n_entities - num_items
            if num_other_entities < 0:
                 raise ValueError("n_entities cannot be smaller than n_items from pretrain_data")

            # Initialize other entities only if there are any
            if num_other_entities > 0:
                other_embed = initializer([num_other_entities, self.emb_dim])
                entity_init_value = tf.concat([item_embed, other_embed], 0)
            else: # If n_entities == n_items
                entity_init_value = item_embed

            all_weights['entity_embed'] = tf_v1.Variable(initial_value=entity_init_value,
                                                      trainable=True, name='entity_embed', dtype=tf.float32)

            print('using pretrained initialization')

        # TF1_COMPAT: Use v1 Variable
        all_weights['relation_embed'] = tf_v1.Variable(initializer([self.n_relations, self.kge_dim]),
                                                    name='relation_embed')
        all_weights['trans_W'] = tf_v1.Variable(initializer([self.n_relations, self.emb_dim, self.kge_dim]), name='trans_W') # For TransR style projection

        self.weight_size_list = [self.emb_dim] + self.weight_size


        for k in range(self.n_layers):
             # TF1_COMPAT: Use v1 Variable
            all_weights['W_gc_%d' %k] = tf_v1.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_gc_%d' % k)
            all_weights['b_gc_%d' %k] = tf_v1.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_gc_%d' % k)

            all_weights['W_bi_%d' % k] = tf_v1.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='W_bi_%d' % k)
            all_weights['b_bi_%d' % k] = tf_v1.Variable(
                initializer([1, self.weight_size_list[k + 1]]), name='b_bi_%d' % k)

            # Weights for GraphSage MLP aggregation if used
            all_weights['W_mlp_%d' % k] = tf_v1.Variable(
                initializer([2 * self.weight_size_list[k], self.weight_size_list[k+1]]), name='W_mlp_%d' % k)
            all_weights['b_mlp_%d' % k] = tf_v1.Variable(
                initializer([1, self.weight_size_list[k+1]]), name='b_mlp_%d' % k)


        return all_weights

    def _build_model_phase_I(self):
        """
        Build the graph convolutional network part for user and entity embeddings.
        """
        if self.alg_type in ['bi', 'kgat']: # Assuming 'kgat' uses 'bi' structure
            self.ua_embeddings, self.ea_embeddings = self._create_bi_interaction_embed()

        elif self.alg_type in ['gcn']:
            self.ua_embeddings, self.ea_embeddings = self._create_gcn_embed()

        elif self.alg_type in ['graphsage']:
            self.ua_embeddings, self.ea_embeddings = self._create_graphsage_embed()
        else:
            print(f'ERROR: Unsupported alg_type {self.alg_type}')
            raise NotImplementedError

        # TF1_COMPAT: Use v1 embedding lookup
        self.u_e = tf_v1.nn.embedding_lookup(self.ua_embeddings, self.users)
        self.pos_i_e = tf_v1.nn.embedding_lookup(self.ea_embeddings, self.pos_items)
        self.neg_i_e = tf_v1.nn.embedding_lookup(self.ea_embeddings, self.neg_items)

        # Used for evaluation, calculating scores for all items for test users
        # This might be memory intensive for large number of items
        # Consider alternative evaluation methods if needed
        # self.batch_predictions = tf.matmul(self.u_e, self.ea_embeddings, transpose_a=False, transpose_b=True)
        # Let's redefine batch_predictions based on how test() function uses it later.
        # If test() computes scores individually, this might not be needed here.
        # Assuming test() calculates scores for test users against a subset of items or all items:
        self.batch_test_scores = tf.matmul(self.u_e, self.ea_embeddings, transpose_a=False, transpose_b=True)


    def _build_model_phase_II(self):
        """
        Build the Knowledge Graph Embedding (KGE) part.
        """
        self.h_e, self.r_e, self.pos_t_e, self.neg_t_e = self._get_kg_inference(self.h, self.r, self.pos_t, self.neg_t)
        # Score for calculating attention weights (based on TransE-like interaction)
        self.A_kg_score = self._generate_kg_score(h=self.h, t=self.pos_t, r=self.r)
        # Create the sparse attention matrix using the calculated scores
        self.A_out = self._create_attentive_A_out()


    def _get_kg_inference(self, h, r, pos_t, neg_t):
        """
        Performs projection for head and tail entities using relation-specific matrices (TransR style).
        """
        # TF1_COMPAT: Use v1 embedding lookup
        embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        # embeddings = tf.expand_dims(embeddings, 1) # This expand_dims might be incorrect for TransR lookup

        # head & tail entity embeddings: batch_size * emb_dim
        h_e_orig = tf_v1.nn.embedding_lookup(embeddings, h)
        pos_t_e_orig = tf_v1.nn.embedding_lookup(embeddings, pos_t)
        neg_t_e_orig = tf_v1.nn.embedding_lookup(embeddings, neg_t)

        # relation embeddings: batch_size * kge_dim
        r_e = tf_v1.nn.embedding_lookup(self.weights['relation_embed'], r)

        # relation transform weights: n_relations * emb_dim * kge_dim -> batch_size * emb_dim * kge_dim
        # TF1_COMPAT: Use v1 embedding lookup
        trans_M = tf_v1.nn.embedding_lookup(self.weights['trans_W'], r)

        # Project head and tail entities to relation-specific space
        # Reshape entities to [batch_size, 1, emb_dim] for bmm with trans_M [batch_size, emb_dim, kge_dim]
        h_e = tf.matmul(tf.expand_dims(h_e_orig, 1), trans_M) # Shape: [batch_size, 1, kge_dim]
        pos_t_e = tf.matmul(tf.expand_dims(pos_t_e_orig, 1), trans_M) # Shape: [batch_size, 1, kge_dim]
        neg_t_e = tf.matmul(tf.expand_dims(neg_t_e_orig, 1), trans_M) # Shape: [batch_size, 1, kge_dim]

        # Reshape back to [batch_size, kge_dim]
        h_e = tf.reshape(h_e, [-1, self.kge_dim])
        pos_t_e = tf.reshape(pos_t_e, [-1, self.kge_dim])
        neg_t_e = tf.reshape(neg_t_e, [-1, self.kge_dim])


        # Original KGAT paper did not normalize relation embeddings in TransR part
        # h_e = tf.math.l2_normalize(h_e, axis=1)
        # r_e = tf.math.l2_normalize(r_e, axis=1) # Relation embed usually not normalized here
        # pos_t_e = tf.math.l2_normalize(pos_t_e, axis=1)
        # neg_t_e = tf.math.l2_normalize(neg_t_e, axis=1)

        return h_e, r_e, pos_t_e, neg_t_e

    def _build_loss_phase_I(self):
        """
        Build the BPR loss for the recommendation part.
        """
        pos_scores = tf.reduce_sum(tf.multiply(self.u_e, self.pos_i_e), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(self.u_e, self.neg_i_e), axis=1)

        # L2 regularization for embeddings involved in the CF loss
        regularizer = tf.nn.l2_loss(self.u_e) + tf.nn.l2_loss(self.pos_i_e) + tf.nn.l2_loss(self.neg_i_e)
        # Normalize regularization term by batch size
        regularizer = regularizer / tf.cast(self.batch_size, tf.float32) # Cast batch_size

        # BPR loss using softplus for numerical stability
        base_loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        # Original BPR log-sigmoid:
        # maxi = tf.log(tf.nn.sigmoid(pos_scores - neg_scores))
        # base_loss = tf.negative(tf.reduce_mean(maxi))

        self.base_loss = base_loss
        self.kge_loss = tf.constant(0.0, tf.float32) # KGE loss is handled in phase II
        self.reg_loss = self.regs[0] * regularizer # Use first reg coeff for CF part
        self.loss = self.base_loss + self.reg_loss # Only CF loss + Reg loss in this phase optimizer

        # Optimization process.
        # TF1_COMPAT: Use v1 AdamOptimizer
        self.opt = tf_v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _build_loss_phase_II(self):
        """
        Build the BPR loss for the KGE part (TransR).
        """
        # Define the scoring function for TransR
        def _get_kg_score(h_e, r_e, t_e):
            # TransR score: || h_proj + r - t_proj ||^2
            kg_score = tf.reduce_sum(tf.square((h_e + r_e - t_e)), axis=1, keepdims=True) # Use axis=1
            return kg_score

        pos_kg_score = _get_kg_score(self.h_e, self.r_e, self.pos_t_e)
        neg_kg_score = _get_kg_score(self.h_e, self.r_e, self.neg_t_e) # Use neg_t_e

        # BPR loss for KGE: We want pos_kg_score to be smaller than neg_kg_score
        # Loss = -log(sigmoid(neg_kg_score - pos_kg_score))
        # Using softplus for stability: softplus(pos_kg_score - neg_kg_score)
        kg_loss = tf.reduce_mean(tf.nn.softplus(pos_kg_score - neg_kg_score)) # Swapped order inside softplus for minimization
        # Original BPR log-sigmoid:
        # maxi = tf.log(tf.nn.sigmoid(neg_kg_score - pos_kg_score))
        # kg_loss = tf.negative(tf.reduce_mean(maxi))


        # L2 regularization for embeddings involved in the KGE loss
        kg_reg_loss = tf.nn.l2_loss(self.h_e) + tf.nn.l2_loss(self.r_e) + \
                      tf.nn.l2_loss(self.pos_t_e) + tf.nn.l2_loss(self.neg_t_e)
        # Normalize regularization term by KG batch size
        kg_reg_loss = kg_reg_loss / tf.cast(self.batch_size_kg, tf.float32) # Cast batch_size_kg

        self.kge_loss2 = kg_loss
        self.reg_loss2 = self.regs[1] * kg_reg_loss # Use second reg coeff for KGE part
        self.loss2 = self.kge_loss2 + self.reg_loss2

        # Optimization process.
        # TF1_COMPAT: Use v1 AdamOptimizer
        self.opt2 = tf_v1.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss2)

    def _create_bi_interaction_embed(self):
        """
        Create user/entity representations based on Bi-Interaction pooling (sum + element-wise product).
        """
        # Check if A_in is properly initialized
        if self.A_in is None:
            raise ValueError("self.A_in is None. Adjacency matrix must be provided.")

        # TF1_COMPAT: Use v1 SparseTensor conversion if needed, or ensure A_in is already compatible
        # A = self._convert_sp_mat_to_sp_tensor(self.A_in) # Assuming A_in is scipy sparse
        # Generate a set of adjacency sub-matrix views (for efficient processing if needed)
        A_fold_hat = self._split_A_hat(self.A_in) # This returns list of SparseTensor objects

        ego_embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            # Message Passing for layer k

            # Calculate sum of neighbors' embeddings (Equation 3 in KGAT paper without attention for now)
            temp_embed = []
            for f in range(self.n_fold):
                 # TF1_COMPAT: Use v1 sparse matmul
                temp_embed.append(tf_v1.sparse_tensor_dense_matmul(A_fold_hat[f], all_embeddings[k])) # Use embeddings from previous layer

            # Sum messages of neighbors.
            side_embeddings = tf.concat(temp_embed, 0)

            # --- Bi-Interaction Aggregator ---
            # 1. Sum aggregation (Equation 3)
            sum_embeddings = side_embeddings
            # 2. Element-wise product aggregation (Equation 4)
            bi_embeddings = tf.multiply(all_embeddings[k], side_embeddings) # Multiply ego embedding with aggregated neighbors

            # 3. Transformations (Equations 3 & 4 apply W1, W2)
            sum_embeddings_transformed = tf.nn.leaky_relu(
                tf.matmul(sum_embeddings, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])
            bi_embeddings_transformed = tf.nn.leaky_relu(
                tf.matmul(bi_embeddings, self.weights['W_bi_%d' % k]) + self.weights['b_bi_%d' % k])

            # 4. Combine aggregations (Equation 5)
            ego_embeddings_new = sum_embeddings_transformed + bi_embeddings_transformed

            # 5. Message dropout.
             # TF1_COMPAT: Use v1 dropout
            ego_embeddings_new = tf_v1.nn.dropout(ego_embeddings_new, rate=self.mess_dropout[k]) # rate is keep_prob complement

            # 6. Normalize the embeddings.
            norm_embeddings = tf.math.l2_normalize(ego_embeddings_new, axis=1)

            all_embeddings.append(norm_embeddings) # Append the new layer's embeddings

        # Concatenate embeddings from all layers (Equation 6)
        all_embeddings_final = tf.concat(all_embeddings, axis=1)

        ua_embeddings, ea_embeddings = tf.split(all_embeddings_final, [self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings


    def _create_gcn_embed(self):
        """
        Create user/entity representations based on GCN.
        """
        if self.A_in is None: raise ValueError("self.A_in is None.")
        # A = self._convert_sp_mat_to_sp_tensor(self.A_in)
        A_fold_hat = self._split_A_hat(self.A_in)

        embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        all_embeddings = [embeddings]

        for k in range(0, self.n_layers):
            # GCN aggregation: A * E * W
            temp_embed = []
            for f in range(self.n_fold):
                 # TF1_COMPAT: Use v1 sparse matmul
                temp_embed.append(tf_v1.sparse_tensor_dense_matmul(A_fold_hat[f], all_embeddings[k]))
            embeddings_aggregated = tf.concat(temp_embed, 0)

            # Transformation
            embeddings_new = tf.nn.leaky_relu(
                tf.matmul(embeddings_aggregated, self.weights['W_gc_%d' % k]) + self.weights['b_gc_%d' % k])

            # Dropout
             # TF1_COMPAT: Use v1 dropout
            embeddings_new = tf_v1.nn.dropout(embeddings_new, rate=self.mess_dropout[k]) # rate is keep_prob complement

            # Normalize the distribution of embeddings.
            norm_embeddings = tf.math.l2_normalize(embeddings_new, axis=1)

            all_embeddings.append(norm_embeddings)

        # Concatenate embeddings from all layers
        all_embeddings_final = tf.concat(all_embeddings, axis=1)

        ua_embeddings, ea_embeddings = tf.split(all_embeddings_final, [self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings

    def _create_graphsage_embed(self):
        """
        Create user/entity representations based on GraphSAGE (mean aggregator).
        """
        if self.A_in is None: raise ValueError("self.A_in is None.")
        # A = self._convert_sp_mat_to_sp_tensor(self.A_in) # Adjacency matrix (potentially normalized)
        A_fold_hat = self._split_A_hat(self.A_in)

        pre_embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        all_embeddings = [pre_embeddings] # Keep initial embeddings

        for k in range(self.n_layers):
            # 1. Aggregate neighbor features (mean aggregator in this case)
            temp_embed = []
            for f in range(self.n_fold):
                 # TF1_COMPAT: Use v1 sparse matmul
                temp_embed.append(tf_v1.sparse_tensor_dense_matmul(A_fold_hat[f], all_embeddings[k])) # Use previous layer's embeddings
            aggregated_neighbors = tf.concat(temp_embed, 0)

            # 2. Concatenate with current node's embedding (from previous layer)
            concat_embeddings = tf.concat([all_embeddings[k], aggregated_neighbors], axis=1)

            # 3. Apply MLP transformation (W_mlp) and activation
            embeddings_new = tf.nn.relu( # GraphSAGE often uses ReLU
                tf.matmul(concat_embeddings, self.weights['W_mlp_%d' % k]) + self.weights['b_mlp_%d' % k])

            # 4. Dropout
             # TF1_COMPAT: Use v1 dropout
            embeddings_new = tf_v1.nn.dropout(embeddings_new, rate=self.mess_dropout[k]) # rate is keep_prob complement

            # 5. Normalize the embeddings (important for stable training)
            norm_embeddings = tf.math.l2_normalize(embeddings_new, axis=1)

            all_embeddings.append(norm_embeddings) # Add normalized embeddings for this layer

        # Concatenate embeddings from all layers (including initial)
        all_embeddings_final = tf.concat(all_embeddings, axis=1)

        ua_embeddings, ea_embeddings = tf.split(all_embeddings_final, [self.n_users, self.n_entities], 0)
        return ua_embeddings, ea_embeddings

    def _split_A_hat(self, X):
        """
        Splits the adjacency matrix X into folds for potentially more efficient processing.
        Returns a list of tf.SparseTensor objects.
        """
        A_fold_hat = []
        if self.n_fold <= 0: # Handle case where n_fold is not positive
             print("Warning: n_fold is not positive. Using the whole matrix.")
             return [self._convert_sp_mat_to_sp_tensor(X)]


        fold_len = (self.n_users + self.n_entities) // self.n_fold
        if fold_len == 0: # Avoid division by zero or empty folds if matrix is smaller than n_fold
            print("Warning: Matrix size smaller than n_fold. Using n_fold=1.")
            self.n_fold = 1
            fold_len = self.n_users + self.n_entities

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_entities
            else:
                end = (i_fold + 1) * fold_len

            # Ensure X is a scipy sparse matrix before slicing and converting
            if not isinstance(X, sp.spmatrix):
                raise TypeError("Input X must be a SciPy sparse matrix for slicing.")

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        Converts a SciPy sparse matrix to a tf.SparseTensor.
        """
        if not isinstance(X, sp.spmatrix):
             raise TypeError("Input X must be a SciPy sparse matrix.")

        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        # TF1_COMPAT: Use v1 SparseTensor
        return tf_v1.SparseTensor(indices, coo.data, coo.shape)


    def _create_attentive_A_out(self):
        """
        Creates the attention-weighted adjacency matrix using pre-calculated KG scores.
        """
        # Assume self.all_h_list, self.all_t_list contain the indices (rows, cols)
        # and self.A_values (placeholder) will be fed with the calculated attention scores (self.A_kg_score)
        indices = np.mat([self.all_h_list, self.all_t_list]).transpose()

        # Create a SparseTensor with the attention scores fed via placeholder
         # TF1_COMPAT: Use v1 SparseTensor
        A_sparse = tf_v1.SparseTensor(indices, self.A_values, self.A_in.shape)

        # Apply sparse softmax to normalize attention scores per node (row-wise)
         # TF1_COMPAT: Use v1 sparse softmax
        A_out = tf_v1.sparse.softmax(A_sparse)
        return A_out

    def _generate_kg_score(self, h, t, r):
        """
        Calculates the KG triple score (e.g., based on TransR) used for attention.
        This should match the KGE scoring function used for optimization if applicable.
        """
        # Get projected embeddings similar to _get_kg_inference
        # TF1_COMPAT: Use v1 embedding lookup
        embeddings = tf.concat([self.weights['user_embed'], self.weights['entity_embed']], axis=0)
        h_e_orig = tf_v1.nn.embedding_lookup(embeddings, h)
        t_e_orig = tf_v1.nn.embedding_lookup(embeddings, t)
        r_e = tf_v1.nn.embedding_lookup(self.weights['relation_embed'], r)
        trans_M = tf_v1.nn.embedding_lookup(self.weights['trans_W'], r)

        h_e = tf.matmul(tf.expand_dims(h_e_orig, 1), trans_M)
        t_e = tf.matmul(tf.expand_dims(t_e_orig, 1), trans_M)

        h_e = tf.reshape(h_e, [-1, self.kge_dim])
        t_e = tf.reshape(t_e, [-1, self.kge_dim])

        # Calculate score - Example using TransE-like score within the projected space for attention
        # This score is used for attention weights, not necessarily the KGE training loss score
        # Original KGAT uses score = t_e^T * tanh(h_e + r_e)
        kg_score = tf.reduce_sum(tf.multiply(t_e, tf.tanh(h_e + r_e)), axis=1) # Keep dimension as [batch_size]

        return kg_score


    def _statistics_params(self):
            # number of params
            total_parameters = 0
            # TF1_COMPAT: Use v1 trainable_variables
            for variable in tf_v1.trainable_variables():
                shape = variable.get_shape()  # shape is a tf.TensorShape object
                variable_parameters = 1
                # TF1_COMPAT FIX: Iterate through shape dimensions
                # In TF2 compat mode, 'dim' might be an int directly, not a Dimension object
                for dim in shape.as_list(): # Use as_list() to get Python list of ints/None
                    if dim is not None:
                        variable_parameters *= dim # Use dim directly, not dim.value
                    else:
                        # Handle unknown dimensions if necessary, e.g., print a warning or skip
                        # print(f"Warning: Found unknown dimension in variable {variable.name}")
                        pass # Or assign a default size if meaningful, though usually skip/warn
                total_parameters += variable_parameters
            if self.verbose > 0:
                print("#params: %d" % total_parameters)

    def train(self, sess, feed_dict):
        # Runs optimization op for phase I (CF part)
        return sess.run([self.opt, self.loss, self.base_loss, self.kge_loss, self.reg_loss], feed_dict)

    def train_A(self, sess, feed_dict):
         # Runs optimization op for phase II (KGE part)
        return sess.run([self.opt2, self.loss2, self.kge_loss2, self.reg_loss2], feed_dict)

    def eval(self, sess, feed_dict):
        # Evaluates user scores against all items
        # TF1_COMPAT: Ensure feed_dict provides necessary inputs (users, dropout rates)
        # Example feed_dict for eval: {self.users: user_batch, self.node_dropout: [0.], self.mess_dropout: [0.] * self.n_layers}
        batch_scores = sess.run(self.batch_test_scores, feed_dict) # Use the scores defined in _build_model_phase_I
        return batch_scores


    def update_attentive_A(self, sess):
        """
        Calculates attention scores for all KG triples and updates the sparse adjacency matrix A_in.
        """
        if self.n_fold <= 0: # Handle case where n_fold is not positive
            print("Warning: n_fold is not positive in update_attentive_A. Cannot fold.")
            fold_len = len(self.all_h_list)
            self.n_fold = 1 # Process as one fold
        else:
            fold_len = len(self.all_h_list) // self.n_fold

        if fold_len == 0 and len(self.all_h_list) > 0: # Handle case where list is smaller than n_fold
             print("Warning: KG list length smaller than n_fold. Using n_fold=1.")
             self.n_fold = 1
             fold_len = len(self.all_h_list)

        kg_score = [] # Store calculated scores

        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = len(self.all_h_list)
            else:
                end = (i_fold + 1) * fold_len

            # Check if slice indices are valid
            if start >= len(self.all_h_list):
                 print(f"Warning: Fold start index {start} out of bounds. Skipping fold {i_fold}.")
                 continue
            if start >= end: # Handle empty slice
                 print(f"Warning: Fold {i_fold} resulted in empty slice [{start}:{end}]. Skipping.")
                 continue


            feed_dict = {
                self.h: self.all_h_list[start:end],
                self.r: self.all_r_list[start:end],
                self.pos_t: self.all_t_list[start:end] # Use pos_t for generating score, as in _generate_kg_score
            }
             # Ensure self.A_kg_score tensor is defined correctly before running
            try:
                A_kg_score_batch = sess.run(self.A_kg_score, feed_dict=feed_dict)
                kg_score.extend(list(A_kg_score_batch)) # Use extend for list
            except Exception as e:
                print(f"Error running A_kg_score in update_attentive_A (fold {i_fold}): {e}")
                # Decide how to handle error (e.g., skip fold, use default scores?)
                # For now, let's skip the update if scoring fails
                print("Skipping attentive A update due to error.")
                return


        if not kg_score:
             print("Warning: No KG scores were calculated. Attentive A cannot be updated.")
             return

        kg_score = np.array(kg_score).astype(np.float32) # Ensure correct dtype

        # Run the op to get the sparse tensor with softmax applied
        # Feed the calculated scores into the placeholder self.A_values
        # Ensure self.A_out tensor is defined correctly
        try:
            new_A_sparse_tensor = sess.run(self.A_out, feed_dict={self.A_values: kg_score})
        except tf.errors.InvalidArgumentError as e:
             print(f"Error running A_out op: {e}")
             print("Ensure shape of fed A_values matches placeholder and sparse tensor definition.")
             print(f"Shape of kg_score fed: {kg_score.shape}")
             print(f"Expected shape based on placeholder A_values: {self.A_values.shape}")
             print("Skipping attentive A update.")
             return
        except Exception as e:
             print(f"Unknown error running A_out op: {e}")
             print("Skipping attentive A update.")
             return


        new_A_values = new_A_sparse_tensor.values
        new_A_indices = new_A_sparse_tensor.indices

        # Reconstruct the scipy sparse matrix
        rows = new_A_indices[:, 0]
        cols = new_A_indices[:, 1]
        self.A_in = sp.coo_matrix((new_A_values, (rows, cols)), shape=(self.n_users + self.n_entities,
                                                                       self.n_users + self.n_entities)).tocsr() # Convert to CSR for efficiency if needed

        # Optional: Add self-loops back if needed by the GNN variant (GCN usually needs them)
        if self.alg_type in ['gcn'] or getattr(self, 'add_self_loops', False): # Add a flag if needed
             # Check format before setdiag
             if not isinstance(self.A_in, (sp.csr_matrix, sp.csc_matrix, sp.lil_matrix)):
                  print("Converting A_in to LIL format for setdiag.")
                  self.A_in = self.A_in.tolil()
             try:
                self.A_in.setdiag(1.)
             except TypeError as e:
                print(f"Error setting diagonal (check A_in format): {e}")
             # Convert back to CSR if preferred for matmuls
             self.A_in = self.A_in.tocsr()