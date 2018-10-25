
import os
import numpy as np
import time

entity_set= set()
rel_set = set()
train_set = set()

w_path = "./FB15k_100"
if not os.path.exists(w_path):
    os.makedirs(w_path)

def rt_toTestFile(dir = "data/FB15k/test.txt",testFile ="test.txt" ):
    # print(os.path.join(w_path,testFile))
    wf_test = open(os.path.join(w_path,testFile), 'w')
    with open(dir) as r_test:
        test_data = r_test.readlines()
        test_data = np.asarray(test_data)
        test_data = np.random.permutation(test_data)
        count = 0
        for line in test_data:
            if line =="" or line =='\n':
                continue
            count += 1
            wf_test.write(line)
            htr = line.strip().split()
            entity_set.add(htr[0])
            entity_set.add(htr[1])
            rel_set.add(htr[2])
            if count == 100:
                break
    wf_test.close()

def rv_toValidFile(dir ="data/FB15k/valid.txt",validFile ="valid.txt"):
    wf_valid = open(os.path.join(w_path,validFile), 'w')
    with open(dir) as r_valid:
        valid_data = r_valid.readlines()
        valid_data = np.asarray(valid_data)
        valid_data = np.random.permutation(valid_data)
        count = 0
        for line in valid_data:
            if line =='' or line == '\n':
                continue
            count += 1
            wf_valid.write(line)
            htr = line.strip().split()
            entity_set.add(htr[0])
            entity_set.add(htr[1])
            rel_set.add(htr[2])
            if count == 100:
                break
    wf_valid.close()

def generate_train_data(dir = "data/FB15k/train.txt",trainFile ="train.txt"):
    rt_toTestFile()
    rv_toValidFile()
    print(len(entity_set),len(rel_set))
    entity_dict = dict();rel_dict = dict()
    with open(dir) as train_data:
        for line in train_data:
            htr = line.strip().split()
            if htr[0] not in entity_dict:
                entity_dict[htr[0]] = 0
            else:
                entity_dict[htr[0]] += 1
            if htr[1] not in entity_dict:
                entity_dict[htr[1]] = 0
            else:
                entity_dict[htr[1]] += 1
            if htr[2] not in rel_dict:
                rel_dict[htr[2]] = 0
            else:
                rel_dict[htr[2]] += 1

    with open(dir) as train_data:
        train_data_list = train_data.readlines()
        train_data_list = np.asarray(train_data_list)
        for entity in entity_set:
            entity_triple_n = min(10,entity_dict[entity])
            train_data_list = np.random.permutation(train_data_list)
            c = 0
            for train_t in train_data_list:
                if train_t !='\n' and (entity in train_t.strip().split()):
                    train_set.add(train_t)
                    c += 1
                else: continue
                if c == entity_triple_n:
                    break

        for rel in rel_set:
            rel_triple_n = min(10,rel_dict[rel])
            train_data_list = np.random.permutation(train_data_list)
            c = 0
            for train_t in train_data_list:
                if train_t !='\n' and (rel in train_t.strip().split()):
                    train_set.add(train_t)
                    c += 1
                else: continue
                if c == rel_triple_n:
                    break
    wf_train = open(os.path.join(w_path, trainFile), 'w')
    for train_t in train_set:
        wf_train.write(train_t)

    wf_train.close()

def generate_id(dir = w_path,file1 = "entity2id.txt",file2 = "relation2id.txt"):
    w_entid = open(os.path.join(dir,file1),'w')
    w_relid = open(os.path.join(dir,file2),'w')
    ent_id = dict();rel_id = dict()
    with open(os.path.join(dir,"train.txt")) as t_file:
        htr_list = t_file.readlines()
        try:
            htr_list.remove('\n')
        except:
            print("can't remove")
        i=0;j=0
        for htr_s in htr_list: # type:str
            htr = htr_s.strip().split()
            if htr[0] not in ent_id:
                ent_id[htr[0]] = i
                i += 1
            if htr[1] not in ent_id:
                ent_id[htr[1]] = i
                i += 1
            if htr[2] not in rel_id:
                rel_id[htr[2]] = j
                j += 1
        for ent in ent_id:
            w_entid.write(ent+'\t'+str(ent_id[ent])+'\n')
        for rel in rel_id:
            w_relid.write(rel+'\t'+str(rel_id[rel])+'\n')



if __name__ == "__main__":
    start_t = time.time()
    #generate_train_data()
    generate_id()
    ent_t = time.time()
    print("time consumed: ",ent_t-start_t)
    print("\n\nok")












































