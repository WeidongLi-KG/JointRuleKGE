import tensorflow as tf
import ntn_input
import ntn
import params
import numpy as np
import numpy.matlib
import random
import datetime

def data_to_indexed(data, entities, relations):
    entity_to_index = {entities[i] : i for i in range(len(entities))}
    relation_to_index = {relations[i] : i for i in range(len(relations))}
    indexed_data = [(entity_to_index[data[i][0]], relation_to_index[data[i][1]],\
            entity_to_index[data[i][2]]) for i in range(len(data))]
    return indexed_data

def get_batch(batch_size, data, num_entities, corrupt_size):
    random_indices = random.sample(range(len(data)), batch_size)
    #data[i][0] = e1, data[i][1] = r, data[i][2] = e2, random=e3 (corrupted)
    batch = [(data[i][0], data[i][1], data[i][2], random.randint(0, num_entities-1))\
	for i in random_indices for j in range(corrupt_size)] # [batch*10,4]
    return batch

def split_batch(data_batch, num_relations):
    batches = [[] for i in range(num_relations)]
    for e1,r,e2,e3 in data_batch:
        batches[r].append((e1,e2,e3)) # [?,batch*10,3]
    return batches

def fill_feed_dict(batches, train_both, batch_placeholders, label_placeholders, corrupt_placeholder):
    feed_dict = {corrupt_placeholder: [train_both and np.random.random()>0.5]}
    for i in range(len(batch_placeholders)):
        feed_dict[batch_placeholders[i]] = batches[i] # [batch*10,3]
        feed_dict[label_placeholders[i]] = [[0.0] for j in range(len(batches[i]))] # [batch*10,1]
    return feed_dict

def run_training():
    print("Begin!")
    #python list of (e1, R, e2) for entire training set in string form
    print("Load training data...")
    raw_training_data = ntn_input.load_training_data(params.data_path)
    print("Load entities and relations...")
    entities_list = ntn_input.load_entities(params.data_path)
    relations_list = ntn_input.load_relations(params.data_path)
    #python list of (e1, R, e2) for entire training set in index form
    indexed_training_data = data_to_indexed(raw_training_data, entities_list, relations_list)
    print("Load embeddings...")
    (init_word_embeds, entity_to_wordvec) = ntn_input.load_init_embeds(params.data_path)
    # init_word_embeds size (67447,100), entity_to_wordvec size (38696)
    # entity_to_wordvec -> [[45792], [50003], [19154, 50004], [11403], [7456, 6932], [47896, 50004], [24589], [50005, 50006, 50004], [6551], [12288]]

    num_entities = len(entities_list)
    num_relations = len(relations_list)

    num_iters = params.num_iter
    batch_size = params.batch_size
    corrupt_size = params.corrupt_size
    slice_size = params.slice_size

    with tf.Graph().as_default():
        print("Starting to build graph "+str(datetime.datetime.now()))
        batch_placeholders = [tf.placeholder(tf.int32, shape=(None, 3), name='batch_'+str(i)) for i in range(num_relations)]
        label_placeholders = [tf.placeholder(tf.float32, shape=(None, 1), name='label_'+str(i)) for i in range(num_relations)]

        corrupt_placeholder = tf.placeholder(tf.bool, shape=(1)) #Which of e1 or e2 to corrupt?
        inference = ntn.inference(batch_placeholders, corrupt_placeholder, init_word_embeds, entity_to_wordvec, \
                num_entities, num_relations, slice_size, batch_size, False, label_placeholders)
        # [2,r*batch_predictions]

        loss = ntn.loss(inference, params.regularization)
        training = ntn.training(loss, params.learning_rate)

	    # Create a session for running Ops on the Graph.
        print('1    Here------->>>>>>>> run init <<<<<<<<------------')
        sess = tf.Session()

        # Run the Op to initialize the variables.
        init = tf.global_variables_initializer()
        print('2    Here------->>>>>>>> run init <<<<<<<<------------')
        sess.run(init)
        print('run init')
        saver = tf.train.Saver(tf.trainable_variables())
        for i in range(1, num_iters):
            print("Starting iter "+str(i)+" "+str(datetime.datetime.now()))
            data_batch = get_batch(batch_size, indexed_training_data, num_entities, corrupt_size) # [batch*10,4]
            relation_batches = split_batch(data_batch, num_relations) # [num_relations,batch*10,3]

            if i % params.save_per_iter == 0:
                saver.save(sess, params.output_path+"/"+params.data_name+str(i)+'.sess')

            feed_dict = fill_feed_dict(relation_batches, params.train_both, batch_placeholders, label_placeholders, corrupt_placeholder)

            _, loss_value = sess.run([training, loss], feed_dict=feed_dict)
            print('loss_'+str(i)+': ',loss_value)

            #TODO: Eval against dev set?

def main(argv):
    run_training()

if __name__=="__main__":
    tf.app.run()
