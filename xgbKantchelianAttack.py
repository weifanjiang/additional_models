import pprint
from gurobipy import *
from sklearn.datasets import load_svmlight_file
from scipy import sparse
import numpy as np
import json
import math
import random
import os
import xgboost as xgb
import time
import argparse
import pickle
import tqdm

GUARD_VAL = 2e-7
ROUND_DIGITS = 20

PRINT = False

# Based on https://github.com/chenhongge/RobustTrees/blob/master/xgbKantchelianAttack.py

def sigmoid(x):
        return 1 / (1 + math.exp(-x))

def manual_pred(x, json_file):
    leaf_values = []
    leaf_ids = []
    for item in json_file:
        tree = item.copy()
        while 'leaf' not in tree.keys():
            attribute, threshold, nodeid = tree['split'], tree['split_condition'], tree['nodeid']
            if type(attribute) == str:
                attribute = int(attribute[1:])

            if args['model_type'] == 'xgboost':
                left = np.float32(x[attribute]) < np.float32(threshold)
            else:
                left = np.float64(np.float32(x[attribute])) <= threshold
            if left:
                if tree['children'][0]['nodeid'] == tree['yes']:
                    tree = tree['children'][0].copy()
                elif tree['children'][1]['nodeid'] == tree['yes']:
                    tree = tree['children'][1].copy()
                else:
                    pprint.pprint(tree)
                    print('x[attribute]:',x[attribute])
                    raise ValueError('child not found')
            else:
                if tree['children'][0]['nodeid'] == tree['no']:
                    tree = tree['children'][0].copy()
                elif tree['children'][1]['nodeid'] == tree['no']:
                    tree = tree['children'][1].copy()
                else:
                    pprint.pprint(tree)
                    print('x[attribute]:',x[attribute])
                    raise ValueError('child not found')
        leaf_values.append(tree['leaf'])
        leaf_ids.append(tree['nodeid'])
    if args['model_type'] == 'sklearn':
        manual_res = np.sum(leaf_values)/float(len(json_file))
    else:
        # xgboost
        manual_res = np.sum(leaf_values, dtype=np.float32)
        manual_res = np.float32(manual_res)
        #manual_res = np.float32(1)/np.add(np.float32(1), np.exp(-np.float32(1) * manual_res))
        #manual_res = 1/np.add(1, np.exp(1 * manual_res))
    return manual_res, leaf_ids


class xgboost_wrapper():
        def __init__(self, model, binary=False):
                self.model = model
                self.binary = binary
                print('binary classification: ',self.binary)

        def maybe_flat(self, input_data):
                if not isinstance(input_data,np.ndarray):
                        #print(type(input_data))
                        input_data = np.copy(input_data.numpy())
                shape = input_data.shape
                if len(input_data.shape) == 1:
                        input_data = np.copy(input_data[np.newaxis,:])
                if len(input_data.shape) >= 3:
                        input_data = np.copy(input_data.reshape(shape[0],np.prod(shape[1:])))
                return input_data, shape

        def predict(self, input_data):
                input_data, _ = self.maybe_flat(input_data)
                ori_input = np.copy(input_data)
                np.clip(input_data, 0, 1, input_data)
                input_data = xgb.DMatrix(input_data)
                ori_input = xgb.DMatrix(sparse.csr_matrix(ori_input))
                test_predict = np.array(self.model.predict(input_data))
                if self.binary:
                        test_predict = (test_predict > 0.5).astype(int)
                else:
                        test_predict = test_predict.astype(int)
                return test_predict

        def predict_logits(self, input_data):
                input_data, _ = self.maybe_flat(input_data)
                input_back = np.copy(input_data)
                input_data = sparse.csr_matrix(input_data)
                input_data = xgb.DMatrix(input_data)
                test_predict = np.array(self.model.predict(input_data))
                return test_predict

        def predict_leaf(self, input_data):
                input_data, _ = self.maybe_flat(input_data)
                input_back = np.copy(input_data)
                input_data = sparse.csr_matrix(input_data)
                input_data = xgb.DMatrix(input_data)
                test_predict = np.array(self.model.predict(input_data, pred_leaf=True))
                return test_predict

        def predict_label(self, input_data):
                return self.predict(input_data)



class node_wrapper(object):

        def __init__(self, treeid, nodeid, attribute, threshold, left_leaves, right_leaves, root=False):
                # left_leaves and right_leaves are the lists of leaf indices in self.leaf_v_list
                self.attribute = attribute
                self.threshold = threshold
                self.node_pos = []
                self.leaves_lists = []
                self.add_leaves(treeid, nodeid, left_leaves,right_leaves,root)

        #def print(self):
        #       print('node_pos{}, attr:{}, th:{}, leaves:{}'.format(self.node_pos, self.attribute, self.threshold, self.leaves_lists))

        def add_leaves(self, treeid,  nodeid, left_leaves, right_leaves, root=False):
                self.node_pos.append({'treeid':treeid, 'nodeid':nodeid})
                if root:
                        self.leaves_lists.append((left_leaves,right_leaves,'root'))
                else:
                        self.leaves_lists.append((left_leaves,right_leaves))

        def add_grb_var(self, node_grb_var, leaf_grb_var_list):
                self.p_grb_var = node_grb_var
                self.l_grb_var_list = []
                for item in self.leaves_lists:
                        left_leaf_grb_var = [leaf_grb_var_list[i] for i in item[0]]
                        right_leaf_grb_var = [leaf_grb_var_list[i] for i in item[1]]
                        if len(item) == 3:
                                self.l_grb_var_list.append((left_leaf_grb_var,right_leaf_grb_var,'root'))
                        else:

                                self.l_grb_var_list.append((left_leaf_grb_var,right_leaf_grb_var))



class xgbKantchelianAttack(object):

        def __init__(self, model, order=np.inf, guard_val=GUARD_VAL, round_digits=ROUND_DIGITS, LP=False, binary=True, pos_json_input=None, neg_json_input=None):
                self.LP = LP
                self.binary = binary or (pos_json_input == None) or (neg_json_input == None)
                print('binary: ', self.binary)
                if LP:
                        print('USING LINEAR PROGRAMMING APPROXIMATION!!')
                else:
                        print('USING MILP EXACT SOLVER!!')
                self.guard_val = guard_val
                self.round_digits = round_digits
                print('order is:',order)
                # print('round features to {} digits'.format(self.round_digits))
                print('guard value is :', guard_val)
                print('feature values are rounded to {} digits'.format(round_digits))

                self.model = model
                if args['model_type'] == 'sklearn':
                    # binary class
                    with open(args['model_json'], 'r') as f:
                        self.json_file = json.load(f)
                        if type(self.json_file) is not list:
                            raise ValueError('model input should be a list of dict loaded by json')
                        print('number of trees:',len(self.json_file))
                    # ignore multiclass for sklearn for now.
                else:
                    # here model is a mnist_xgboost_wrapper model
                    #self.model = model
                    if self.binary :
                            temp = 'temporary{}.json'.format(str(round(time.time() * 1000))[-4:])
                            print('temp file:', temp)
                            model.model.dump_model(temp,dump_format='json')
                            with open(temp) as f:
                                    self.json_file = json.load(f)
                            if type(self.json_file) is not list:
                                    raise ValueError('model input should be a list of dict loaded by json')
                            else:
                                    os.remove(temp)
                            print('number of trees:',len(self.json_file))
                    else:
                            self.pos_json_file = pos_json_input
                            self.neg_json_file = neg_json_input
                            print('number of pos trees:',len(self.pos_json_file))
                            print('number of neg trees:',len(self.neg_json_file))

                # save number of trees
                if args['model_type'] == 'sklearn':
                    self.n_trees = len(self.json_file)

                self.order = order
                # two nodes with identical decision are merged in this list, their left and right leaves and in the list, third element of the tuple
                self.node_list = []
                self.leaf_v_list = [] # list of all leaf values
                self.leaf_pos_list = [] # list of leaves' position in xgboost model
                self.leaf_count = [0] # total number of leaves in the first i trees
                node_check = {} # track identical decision nodes. {(attr, th):<index in node_list>}

                def dfs(tree, treeid, root=False, neg=False):
                        if 'leaf' in tree.keys():

                                leaf_val = tree['leaf']
                                '''
                                if args['model_type'] == 'sklearn':
                                    leaf_val = np.float64(np.float32(leaf_val))
                                '''

                                if neg:
                                        self.leaf_v_list.append(-1 * leaf_val)
                                else:
                                        self.leaf_v_list.append(leaf_val)
                                self.leaf_pos_list.append({'treeid':treeid,'nodeid':tree['nodeid']})
                                return [len(self.leaf_v_list)-1]
                        else:
                                attribute, threshold, nodeid = tree['split'], tree['split_condition'], tree['nodeid']
                                if type(attribute) == str:
                                        attribute = int(attribute[1:])
                                # threshold = round(threshold, self.round_digits)

                                if args['model_type'] == 'xgboost':
                                    threshold = np.float64(str(np.float32(threshold)))

                                left_subtree = None
                                right_subtree = None
                                for subtree in tree['children']:
                                        if subtree['nodeid'] == tree['yes']:
                                                left_subtree = subtree
                                        if subtree['nodeid'] == tree['no']:
                                                right_subtree = subtree
                                if left_subtree == None or right_subtree == None:
                                        pprint.pprint(tree)
                                        raise ValueError('should be a tree but one child is missing')
                                left_leaves = dfs(left_subtree, treeid, False, neg)
                                right_leaves = dfs(right_subtree, treeid, False, neg)
                                if (attribute, threshold) not in node_check:
                                        self.node_list.append(node_wrapper(treeid, nodeid, attribute, threshold, left_leaves, right_leaves, root))
                                        node_check[(attribute,threshold)] = len(self.node_list)-1
                                else:
                                        node_index = node_check[(attribute, threshold)]
                                        self.node_list[node_index].add_leaves(treeid, nodeid, left_leaves,right_leaves,root)
                                return left_leaves + right_leaves

                if self.binary:
                        for i,tree in enumerate(self.json_file):
                                dfs(tree,i, root=True)
                                self.leaf_count.append(len(self.leaf_v_list))
                        if len(self.json_file)+1 != len(self.leaf_count):
                                print('self.leaf_count:',self.leaf_count)
                                raise ValueError('leaf count error')
                else:
                        for i,tree in enumerate(self.pos_json_file):
                                dfs(tree, i, root=True)
                                self.leaf_count.append(len(self.leaf_v_list))
                        for i,tree in enumerate(self.neg_json_file):
                                dfs(tree, i+len(self.pos_json_file), root=True, neg=True)
                                self.leaf_count.append(len(self.leaf_v_list))
                        if len(self.pos_json_file)+len(self.neg_json_file)+1 != len(self.leaf_count):
                                print('self.leaf_count:',self.leaf_count)
                                raise ValueError('leaf count error')

                self.m = Model('attack')
                self.m.setParam('Threads', args['threads'])
                if self.LP:
                        self.P = self.m.addVars(len(self.node_list), lb=0, ub=1, name='p')
                else:
                        self.P = self.m.addVars(len(self.node_list), vtype=GRB.BINARY, name='p')
                self.L = self.m.addVars(len(self.leaf_v_list), lb=0, ub=1, name='l')
                if self.order == np.inf:
                        self.B = self.m.addVar(name='b')
                self.llist = [self.L[key] for key in range(len(self.L))]
                self.plist = [self.P[key] for key in range(len(self.P))]

                '''
                print('leaf value list:',self.leaf_v_list)
                print('number of leaves in the first k trees:',self.leaf_count)
                '''

                # p dictionary by attributes, {attr1:[(threshold1, gurobiVar1),(threshold2, gurobiVar2),...],attr2:[...]}
                self.pdict = {}
                for i,node in enumerate(self.node_list):
                        node.add_grb_var(self.plist[i], self.llist)

                        split_val = node.threshold

                        if node.attribute not in self.pdict:
                                self.pdict[node.attribute] = [(split_val,self.plist[i])]
                        else:
                                self.pdict[node.attribute].append((split_val,self.plist[i]))

                # sort each feature list
                # add p constraints
                for key in self.pdict.keys():
                        min_diff = 1000
                        if len(self.pdict[key])>1:
                                self.pdict[key].sort(key=lambda tup: tup[0])
                                for i in range(len(self.pdict[key])-1):
                                        self.m.addConstr(self.pdict[key][i][1]<=self.pdict[key][i+1][1], name='p_consis_attr{}_{}th'.format(key,i))
                                        min_diff = min( min_diff, self.pdict[key][i+1][0]-self.pdict[key][i][0])
                                # print('attr {} min difference between thresholds:{}'.format(key,min_diff))

                                # additional constraints:
                                # when the difference between two split values are less than the data percision (usually 1e-6)
                                # make sure gurobi does not try to choose a path between the very close splits.
                                data_precision = 10**(-1*args["round_data"])
                                if args["model_type"] == 'sklearn':
                                    for i in range(1, len(self.pdict[key])):

                                        l = self.pdict[key][i-1][0]
                                        r = self.pdict[key][i][0]
                                        feat = r
                                        while np.float32(feat) > np.float64(r):
                                            feat -= self.guard_val
                                        if np.float32(feat) <= l:
                                            self.m.addConstr(self.pdict[key][i][1]==self.pdict[key][i-1][1], name='float32_consis_attr{}_{}th'.format(key,i-1))

                                            '''
                                            print("Additional constraints on {} and {}".format(self.pdict[key][i-1][0],
                                                self.pdict[key][i][0]))
                                            '''

                                if args["model_type"] == 'xgboost':
                                    for i in range(1, len(self.pdict[key])):
                                        if np.absolute(np.float32(self.pdict[key][i][0]) - np.float32(self.pdict[key][i-1][0])) < np.float32(data_precision):
                                            self.m.addConstr(self.pdict[key][i][1]==self.pdict[key][i-1][1], name='float32_consis_attr{}_{}th'.format(key,i-1))

                                # if min_diff < self.guard_val * 10:
                                        # self.guard_val = min_diff/10
                                        # print('guard value too large, change to min_diff/3:',self.guard_val)
                                        # print('guard value too large for feature: ', key)

                        '''
                        if key == 574:
                            all_splits = [x[0] for x in self.pdict[key]]
                            print(all_splits)
                        '''
                        
                
                print('update guard val to: ', self.guard_val)


                # all leaves sum up to 1
                for i in range(len(self.leaf_count)-1):
                        leaf_vars = [self.llist[j] for j in range(self.leaf_count[i],self.leaf_count[i+1])]
                        self.m.addConstr(LinExpr([1]*(self.leaf_count[i+1]-self.leaf_count[i]),leaf_vars)==1, name='leaf_sum_one_for_tree{}'.format(i))

                # node leaves constraints
                for j in range(len(self.node_list)):
                        p = self.plist[j]
                        for k in range(len(self.node_list[j].leaves_lists)):
                                left_l = [self.llist[i] for i in self.node_list[j].leaves_lists[k][0]]
                                right_l = [self.llist[i] for i in self.node_list[j].leaves_lists[k][1]]
                                if len(self.node_list[j].leaves_lists[k]) == 3:
                                        self.m.addConstr(LinExpr([1]*len(left_l), left_l)-p==0, name='p{}_root_left_{}'.format(j,k))
                                        self.m.addConstr(LinExpr([1]*len(right_l), right_l)+p==1, name='p_{}_root_right_{}'.format(j,k))
                                else:
                                        self.m.addConstr(LinExpr([1]*len(left_l), left_l)-p<=0, name='p{}_left_{}'.format(j,k))
                                        self.m.addConstr(LinExpr([1]*len(right_l), right_l)+p<=1, name='p{}_right_{}'.format(j,k))
                self.m.update()


        def attack(self, X, label=None, feature_weight = dict()):
                if args['model_type'] == 'sklearn':
                    pred = self.model.predict(np.array([X]))[0]
                else:
                    pred = self.model.predict(X)
                x = np.copy(X).astype(np.float64)

                print('\n\n==================================')

                if pred != label:
                        print('wrong prediction, no need to attack')
                        return X

                if PRINT:
                    print('X:',x)
                print('label:',label)
                print('prediction:',pred)
                # model mislabel
                # this is for binary
                try:
                        c = self.m.getConstrByName('mislabel')
                        self.m.remove(c)
                        self.m.update()
                except Exception:
                        pass
                
                if (not self.binary) or label == 1:
                    if args['model_type'] == 'sklearn':
                        # average prediction value from all the trees
                        self.m.addConstr(LinExpr(self.leaf_v_list,self.llist)<=0.5*self.n_trees - 1e-4, name='mislabel')
                        #self.m.addConstr(LinExpr(self.leaf_v_list,self.llist)<=0.5*self.n_trees - 0.001, name='mislabel')
                    else:
                        self.m.addConstr(LinExpr(self.leaf_v_list,self.llist)<=0, name='mislabel')
                else:
                    if args['model_type'] == 'sklearn':
                        # average prediction value from all the trees
                        self.m.addConstr(LinExpr(self.leaf_v_list,self.llist)>=0.5*self.n_trees + 1e-4, name='mislabel')
                        #self.m.addConstr(LinExpr(self.leaf_v_list,self.llist)>=0.5*self.n_trees + 0.001, name='mislabel')
                    else:
                        self.m.addConstr(LinExpr(self.leaf_v_list,self.llist)>=self.guard_val, name='mislabel')
                self.m.update()

                if self.order == np.inf:
                        rho = 1
                else:
                        rho = self.order

                if self.order != np.inf:
                        self.obj_coeff_list = []
                        self.obj_var_list = []
                        self.obj_c = 0
                # model objective
                #print(self.pdict)
                #print(x)
                for key in self.pdict.keys():
                        if PRINT: print(x[key])
                        if len(self.pdict[key]) == 0:
                                raise ValueError('self.pdict list empty')
                        axis = [-np.inf] + [item[0] for item in self.pdict[key]] + [np.inf]
                        w = [0] * (len(self.pdict[key])+1)
                        for i in range(len(axis)-1,0,-1):
                                if x[key] < axis[i] and x[key] >= axis[i-1]:
                                        w[i-1] = 0
                                elif x[key] < axis[i] and x[key] < axis[i-1]:
                                        #print("feature_weight[key][1]", key, feature_weight[key][1], rho)
                                        #w[i-1] = np.abs(feature_weight[key][1]*(x[key]-axis[i-1]))**rho
                                        w[i-1] = np.abs(x[key]-axis[i-1])**rho
                                elif x[key] >= axis[i] and x[key] >= axis[i-1]:
                                        #print("feature_weight[key][0]", key, feature_weight[key][0], rho)
                                        w[i-1] = np.abs(x[key]-axis[i]+self.guard_val)**rho
                                else:
                                        print('x[key]:',x[key])
                                        print('axis:',axis)
                                        print('axis[i]:{}, axis[i-1]:{}'.format(axis[i],axis[i-1]))
                                        raise ValueError('wrong axis ordering')
                        for i in range(len(w)-1):
                                w[i] -= w[i+1]
                        if self.order != np.inf:
                                self.obj_c += w[-1]
                                self.obj_coeff_list += w[:-1]
                                self.obj_var_list += [item[1] for item in self.pdict[key]]
                        else:
                                try:
                                        c = self.m.getConstrByName('linf_constr_attr{}'.format(key))
                                        self.m.remove(c)
                                        self.m.update()
                                except Exception:
                                        pass
                                self.m.addConstr(LinExpr(w[:-1],[item[1] for item in self.pdict[key]])+w[-1]<=self.B, name='linf_constr_attr{}'.format(key))
                                self.m.update()

                point_constraints = set()

                # add non-negative feature constraints
                nn_count = 1
                for key in self.pdict.keys():
                        if key >= len(x):
                                continue
                        all_nodes_with_key = self.pdict[key]
                        for i in range(len(all_nodes_with_key)):
                                node_to_check = all_nodes_with_key[i]
                                if (X[key] > node_to_check[0] and node_to_check[0] < self.guard_val):
                                        self.m.addConstr(lhs=node_to_check[1], sense=GRB.LESS_EQUAL, rhs=0, name='non_negative_{}'.format(nn_count))
                                        point_constraints.add('non_negative_{}'.format(nn_count))
                                        self.m.update()
                                        nn_count += 1

                # add less-than-one feature constraints
                lo_count = 1
                for key in self.pdict.keys():
                        if key >= len(x):
                                continue
                        all_nodes_with_key = self.pdict[key]
                        for i in range(len(all_nodes_with_key)):
                                node_to_check = all_nodes_with_key[i]
                                if (X[key] <= node_to_check[0] and node_to_check[0] + self.guard_val > 1):
                                        self.m.addConstr(lhs=node_to_check[1], sense=GRB.GREATER_EQUAL, rhs=1, name='less_than_one_{}'.format(lo_count))
                                        point_constraints.add('less_than_one_{}'.format(lo_count))
                                        self.m.update()
                                        lo_count += 1
                

                if self.order != np.inf:
                        #print("SETTING MINIMIZATION CONSTRAINT")
                        self.m.setObjective(LinExpr(self.obj_coeff_list, self.obj_var_list)+self.obj_c, GRB.MINIMIZE)
                else:
                        self.m.setObjective(self.B, GRB.MINIMIZE)
                self.m.update()
                self.m.optimize()

                # check Optimization Status Code of the model
                if self.m.Status == GRB.OPTIMAL:
                        print('model was optimally solved\n')
                else:
                        print('model was not optimally solved with status code = {}\n'.format(self.m.Status))
                        for c_name in point_constraints:
                                c = self.m.getConstrByName(c_name)
                                self.m.remove(c)
                                self.m.update()
                        return None

                print('Obj: %g' % self.m.objVal)
                if args['model_type'] == 'xgboost':
                    for key in self.pdict.keys():
                            for node in self.pdict[key]:
                                    if node[1].x > 0.5 and x[key] >= node[0]:
                                            x[key] = node[0] - self.guard_val
                                    if node[1].x <= 0.5 and x[key] < node[0]:
                                            x[key] = node[0] + self.guard_val
                else:
                    # sklearn
                    for key in self.pdict.keys():
                            for node in self.pdict[key]:

                                    if node[1].x > 0.5 and np.float32(x[key]) > np.float64(node[0]):
                                            x[key] = node[0]

                                            while np.float32(x[key]) > np.float64(node[0]):
                                                x[key] -= self.guard_val
                                            
                                    if node[1].x <= 0.5 and np.float32(x[key]) <= np.float64(node[0]):
                                            x[key] = node[0]

                                            while np.float32(x[key]) <= np.float64(node[0]):
                                                x[key] += self.guard_val

                # Debug: check whether perturbed input satisfies Gurobi's variables.
                '''                 
                for key in self.pdict.keys():
                    for node in self.pdict[key]:
                        if node[1].x > 0.5 and np.float32(x[key]) > np.float32(node[0]):
                            print("wrong on ", key)
                            print("split is ", node[0])
                            print("GRB solution: ", node[1].x)
                            print("feature val ", x[key], np.float64(np.float32(x[key])))
                            all_splits = [x[0] for x in self.pdict[key]]
                            all_grb = [x[1].x for x in self.pdict[key]]
                            print("all splits & grb for this feature: {}".format(list(zip(all_splits, all_grb))))

                        if node[1].x <= 0.5 and np.float32(x[key]) <= np.float64(node[0]):
                            print("wrong on ", key)
                            print("split is ", node[0])
                            print("GRB solution: ", node[1].x)
                            print("feature val ", x[key], np.float64(np.float32(x[key])))
                            all_splits = [x[0] for x in self.pdict[key]]
                            all_grb = [x[1].x for x in self.pdict[key]]
                            print("all splits & grb for this feature: {}".format(list(zip(all_splits, all_grb))))
                '''

                print('\n-------------------------------------\n')
                # print('result for this point:',x)
                if args['model_type'] == 'sklearn':
                    adv_pred = self.model.predict(np.array([x]))[0]
                else:
                    adv_pred = self.model.predict(x)
                print("adv_pred:", adv_pred, "label:", label)
                if adv_pred != label:
                        suc = True
                else:
                        suc = False
                print('success from solver:', suc)
                print('mislabel constraint:', np.sum(np.array(self.leaf_v_list)*np.array([item.x for item in self.llist])))
                if args["model_type"] == 'sklearn':
                    print('prediction supposed to be:', np.sum(np.array(self.leaf_v_list)*np.array([item.x for item in self.llist]))/self.n_trees)

                    # Debug: print the activated leaves suggested by gurobi.
                    '''
                    gurobi_leaf_ids = np.zeros(self.n_trees)
                    for item, leaf_stat in zip(self.llist, self.leaf_pos_list):
                        if item.x > 0.5:
                            gurobi_leaf_ids[leaf_stat['treeid']] = leaf_stat['nodeid']
                    
                    print("Gurobi's leaf ids: ", gurobi_leaf_ids)
                    '''
                

                nonnegative_feature = np.amin(x) >= 0
                print('no negative features in x\':', nonnegative_feature)
                # Note that the attack counts as "Fail" no matter whether manual prediction succeeds.
                if (not suc):
                        manual_suc = False
                        if self.binary:
                                manual_res, manual_vals, manual_ids = self.check(x, self.json_file)
                        else:
                                manual_res, manual_vals, manual_ids = self.check(x, self.pos_json_file) - self.check(x, self.neg_json_file)
                        print('manual prediction result:', manual_res)
                        if args['model_type'] == 'xgboost':
                            print('pred logits: ', self.model.predict_logits(x))
                            print('pred leaves: ', self.model.predict_leaf(x))
                            if (not self.binary and manual_res>=0)  or (self.binary and int(manual_res>0) == label):
                                    manual_suc = False
                            else:
                                    manual_suc = True
                        else:
                            # sklearn
                            print('model prediction result:', self.model.predict_proba([x])[0])
                            model_leaves = self.model.apply(x.reshape((1, -1)))[0]
                            '''
                            print(manual_ids)
                            print(list(model_leaves))
                            '''
                            bad_count = 0
                            for leaf_id in range(len(manual_ids)):
                                if manual_ids[leaf_id] != model_leaves[leaf_id]:
                                    bad_count += 1
                            print("manual and model have different leaves: ", bad_count)

                            # debug: compare gurobi's activated leaves against manual activated leaves
                            '''
                            bad_count = 0
                            for leaf_id in range(len(manual_ids)):
                                if manual_ids[leaf_id] != gurobi_leaf_ids[leaf_id]:
                                    print("Manual and gurobi different on tree {}".format(leaf_id))
                                    bad_count += 1
                            print("manual and gurobi have different leaves: ", bad_count)
                            '''

                            if (not self.binary and manual_res>=0)  or (self.binary and int(manual_res>0.5) == label):
                                    manual_suc = False
                            else:
                                    manual_suc = True

                for c_name in point_constraints:
                        c = self.m.getConstrByName(c_name)
                        self.m.remove(c)
                self.m.update()

                if suc:
                    return x
                else:
                    if manual_suc:
                        print("WARNING: model prediction failed, but manual prediction succeed.")
                        return np.repeat(np.Infinity, len(X))
                    else:
                        print("WARNING: model/manual prediction failed, but GRB has optimal solution.")
                        return np.repeat(np.Infinity, len(X))

        def check(self, x, json_file):
                # Due to XGBoost precision issues, some attacks may not succeed if tested using model.predict.
                # We manually run the tree on the json file here to make sure those attacks are actually successful.
                print('-------------------------------------\nstart checking')
                print('manually run trees')

                leaf_values = []
                leaf_ids = []
                for item in json_file:
                        tree = item.copy()
                        while 'leaf' not in tree.keys():
                                attribute, threshold, nodeid = tree['split'], tree['split_condition'], tree['nodeid']
                                if type(attribute) == str:
                                        attribute = int(attribute[1:])

                                if args['model_type'] == 'xgboost':
                                    left = np.float32(x[attribute]) < np.float32(threshold)
                                else:
                                    left = np.float32(x[attribute]) <= threshold

                                if left:
                                        if tree['children'][0]['nodeid'] == tree['yes']:
                                                tree = tree['children'][0].copy()
                                        elif tree['children'][1]['nodeid'] == tree['yes']:
                                                tree = tree['children'][1].copy()
                                        else:
                                                pprint.pprint(tree)
                                                print('x[attribute]:',x[attribute])
                                                raise ValueError('child not found')
                                else:
                                        if tree['children'][0]['nodeid'] == tree['no']:
                                                tree = tree['children'][0].copy()
                                        elif tree['children'][1]['nodeid'] == tree['no']:
                                                tree = tree['children'][1].copy()
                                        else:
                                                pprint.pprint(tree)
                                                print('x[attribute]:',x[attribute])
                                                raise ValueError('child not found')
                        leaf_values.append(tree['leaf'])
                        leaf_ids.append(tree['nodeid'])
                if args['model_type'] == 'sklearn':
                    manual_res = np.sum(leaf_values)/float(self.n_trees)
                else:
                    # xgboost
                    leaf_values = np.array(leaf_values, dtype=np.float32)
                    manual_res = np.sum(leaf_values)
                print('sum: {}'.format(manual_res))
                print('leaf ids: {}'.format(leaf_ids))
                return manual_res, leaf_values, leaf_ids



class xgbMultiClassKantchelianAttack(object):
        #def __init__(self, model, num_classes, order=np.inf, guard_val=GUARD_VAL, round_digits=ROUND_DIGITS, LP=False):
        def __init__(self, model, order=1, guard_val=GUARD_VAL, round_digits=ROUND_DIGITS, LP=False, binary=True, pos_json_input=None, neg_json_input=None):
                self.model = model
                self.num_classes = num_classes
                if num_classes <= 2:
                        raise ValueError('multiclass attack must be used when number of class > 2')
                self.order = order
                self.guard_val = guard_val
                self.round_digits = round_digits
                self.LP = LP
                temp = 'temporary{}.json'.format(str(round(time.time() * 1000))[-4:])
                print('temp file: ', temp)
                model.model.dump_model(temp,dump_format='json')
                with open(temp) as f:
                        self.json_file = json.load(f)
                if type(self.json_file) is not list:
                        raise ValueError('model input should be a list of dict loaded by json')
                else:
                        os.remove(temp)
                print('number of trees:',len(self.json_file))
                self.json_inputs = [[] for l in range(self.num_classes)]
                for i,tree in enumerate(self.json_file):
                        self.json_inputs[i%num_classes].append(tree)

        def attack(self, x, label, feature_weight = dict()):
                min_dist = 1e10
                if self.LP:
                        final_adv = x
                else:
                        final_adv = None
                for l in range(self.num_classes):
                        if l == label:
                                continue
                        print('\n************* original label {}, target label {} starts *************'.format(label, l))
                        start_time = time.time()
                        attack_model = xgbKantchelianAttack(self.model, self.order, self.guard_val, self.round_digits, self.LP, binary=False, pos_json_input=self.json_inputs[label[0]], neg_json_input=self.json_inputs[l])
                        end_time = time.time()
                        print('time to build the model: %.4f seconds' % (end_time - start_time))
                        adv = attack_model.attack(x, label, feature_weight)
                        if PRINT:
                        	print('attack result:', adv)
                        if self.LP:
                                if attack_model.m.objVal < min_dist:
                                        min_dist = attack_model.m.objVal
                        else:
                                if adv is None:
                                        print('WARNING!! target label {} has None as output'.format(l))
                                elif np.amin(adv) == np.Infinity:
                                        print('WARNING!! target label {} has Infinity as output'.format(l))
                                else:
                                        adv_pred = self.model.predict(adv)
                                        dist = np.linalg.norm(x-adv,ord=self.order)
                                        suc = adv_pred != label
                                        print('target label {}, adv dist:{}, adv predict:{}, success:{}'.format(l, dist, adv_pred, suc))
                                        if dist < min_dist:
                                                final_adv = adv
                                                min_dist = dist
                if self.LP:
                        print('Final Obj:', min_dist)
                else:
                        final_adv_pred = self.model.predict(final_adv)
                        print('******************************* \nfinal adv dist:{}, final adv predict:{}, success:{}'.format(np.linalg.norm(final_adv-x,ord=self.order), final_adv_pred, final_adv_pred != label))
                return final_adv



def main(args_passin):
        global args
        args = args_passin
        random.seed(8)
        np.random.seed(8)
        binary = (args['num_classes'] == 2)

        with open(args['model_json'], 'r') as fin:
            json_model = json.load(fin)

        if args['model_type'] == 'xgboost':
            bst = xgb.Booster()
            bst.load_model(args['model'])
            model = xgboost_wrapper(bst, binary=binary)
        elif args['model_type'] == 'sklearn':
            model = pickle.load(open(args['model'], 'rb'))
        else:
            return
        
        order = args['order']
        if order == -1:
            order = np.inf

        if binary:
                attack = xgbKantchelianAttack(model, order=order, guard_val=args['guard_val'], round_digits=args['round_digits'])
        else:
                attack = xgbMultiClassKantchelianAttack(model, order=order, num_classes=args['num_classes'], guard_val=args['guard_val'], round_digits=args['round_digits'])

        feature_weight = dict()
        if args['weight'] != 'No weight':
            feature_weight_str_key = json.load(open(args['weight'], 'r'))
            for key, item in feature_weight_str_key.items():
                w_decrease, w_increase = item
                if w_decrease == -1:
                    w_decrease = np.inf
                if w_increase == -1:
                    w_increase = np.inf
                item = [w_decrease, w_increase]
                feature_weight[int(key)] = item
        print(feature_weight)

        # load idx_to_word vocabulary
        # idx_to_word = pickle.load(open('index_to_word.pickle', 'rb'))
        # load test_data from pickle.
        if args['data'].endswith(".pickle"):
            test_data = pickle.load(open(args['data'], 'rb'))
            test_labels = np.ones(test_data.shape[0])
        else:
            test_data, test_labels = load_svmlight_file(args['data'])
        if type(test_data) != np.ndarray:
            test_data = test_data.toarray()
        
        test_data = np.around(test_data, decimals = args["round_data"]).astype(np.float64)

        if "csv" in args["model"] and args["model_type"] == 'xgboost' and "binary_mnist0.t" in args["data"]:
            # The csv binary mnist training data has 785 columns, but the libsvm has 784 columns
            # so append an empty column to the end of libsvm testing data
            row, _ = test_data.shape
            new_data = np.zeros((row, 785), dtype = float)
            new_data[:, 0:784] = test_data
            test_data = new_data

        if args['num_attacks'] == -1:
            args['offset'] = 0
            args['num_attacks'] = test_data.shape[0]
            print("attacking all points.")

        all_adv = []

        if args['model_type'] == 'xgboost' and args['feature_start'] > 0:
                test_data = np.hstack((np.zeros((test_data.shape[0],args['feature_start'])),test_data))
                test_data = np.around(test_data, args['round_data'])
        test_labels = test_labels[:,np.newaxis].astype(int)

        arr = np.arange(test_data.shape[0])
        if not args['no_shuffle']:
            np.random.shuffle(arr)
        samples = arr[args['offset']:args['offset']+args['num_attacks']]
        num_attacks = len(samples) # real number of attacks cannot be larger than test data size
        avg_l0 = 0.
        avg_l1 = 0.
        avg_l2 = 0.
        if args['weight'] != 'No weight':
            avg_l1_weighted = 0.
        avg_linf = 0.
        counter = 0
        label_one_counter = 0
        success_count = 0
        global_start = time.time()
        fout = open(args['out'], 'w')
        fout.write("index\tlabel\tprediction\tl0\tl1\tl2\tlinf\n")

        model_manual_inconsistent = list()
        clean_sample_inconsistent = list()
        milp_non_optimal = list()

        # Before attacking, let's check whether model & json have consistent prediction on all samples.
        if args['num_classes'] == 2 and args['initial_check']:
            all_disagree_sample = list()
            print("total test samples: {}".format(test_data.shape[0]))
            print("Check whether results agree when predicting with model & json...")
            for i in tqdm.tqdm(range(test_data.shape[0])):
                if args['model_type'] == 'sklearn':
                    predict = model.predict(test_data[i:i+1])[0]
                else:
                    predict = model.predict(test_data[i])
                manual_pred_res, _ = manual_pred(test_data[i], json_model)
                if args['model_type'] == 'xgboost':
                    manual_label = int(manual_pred_res > 0)
                else:
                    manual_label = int(manual_pred_res > 0.5)
                if predict != manual_label:
                    all_disagree_sample.append(i)
            print("Model and manual pred disagree on all clean samples: {}".format(all_disagree_sample))
            if len(all_disagree_sample) > 0:
                print("model and manual prediction inconsistent on samples {}, aborting".format(all_disagree_sample))
                exit(0)
            else:
                print("Manual and Model predictions agree on all test samples.")
    

        for n,idx in enumerate(samples):
                print("\n\n\n\n======== Point {} ({}/{}) starts, current avg. linf {} =========".format(idx, n+1, num_attacks, avg_linf/max(1, success_count)))
                if args['model_type'] == 'sklearn':
                    predict = model.predict(test_data[idx:idx+1])[0]
                else:
                    predict = model.predict(test_data[idx])
                
                '''
                # only attack malicious points for security applications
                if test_labels[idx] == 0:
                    print('not malicious, skip.')
                    continue
                '''
                        
                if test_labels[idx] != predict:
                    print('true label:{}, predicted label:{}'.format(test_labels[idx], predict))
                    print('prediction not correct, skip this one.')
                    # fout.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (idx, test_labels[idx][0], predict[0], '0', '0', '0'))
                    continue
                
                manual_pred_res, _ = manual_pred(test_data[idx], json_model)
                if args['model_type'] == 'xgboost':
                    agree = int(manual_pred_res > 0)
                else:
                    agree = int(manual_pred_res > 0.5)
                if predict != agree:
                    print('model pred label:{}, json predicted label:{}'.format(predict, int(manual_pred_res)))
                    print('model & manual pred inconsistent, skip this one.')
                    clean_sample_inconsistent.append(idx)
                    # fout.write('%s\t%s\t%s\t%s\t%s\t%s\n' % (idx, test_labels[idx][0], predict[0], '0', '0', '0'))
                    continue
                
                counter += 1
                if test_labels[idx][0] == 1:
                    label_one_counter += 1

                # adv is None: Liner program is not optimally solved.
                # adv contains all Infinity: model and manual inconsistent predictions
                # adv is a normal vector: attack successful
                adv = attack.attack(test_data[idx], test_labels[idx], feature_weight)

                if adv is None:
                    milp_non_optimal.append(idx)
                    print("MILP no solution, ATTACK failed for point {}".format(idx))
                else:
                    diff = adv-test_data[idx]
                    l0 = np.count_nonzero(diff)
                    l1 = np.linalg.norm(diff, ord=1)
                    l2 = np.linalg.norm(diff)
                    linf = np.max(np.abs(diff))
                    if linf == np.Infinity:
                        model_manual_inconsistent.append(idx)
                        print("Point {}'s attack result has model & manual conflict, count as fail.".format(idx))
                        continue
                    else:
                        ### save the adv example for that sha1
                        all_adv.append(adv)
                        print('adv[0]:', adv[0])
                        if args['feature_start']>0:
                                adv[0] = 0
                                print('0th feature set back to 0. adv[0]:', adv[0])

                        avg_l0 += l0
                        avg_l1 += l1
                        avg_l2 += l2
                        avg_linf += linf
                        success_count += 1

                        if args['weight'] != 'No weight':
                            l1_weighted = 0.
                            for feature_index, item in feature_weight.items():
                                w_decrease, w_increase = item
                                if diff[feature_index] < 0:
                                    l1_weighted += w_decrease*np.abs(diff[feature_index])
                                else:
                                    l1_weighted += w_increase*np.abs(diff[feature_index])

                        if args['weight'] != 'No weight':
                            avg_l1_weighted += l1_weighted
                        
                        if args['model_type'] == 'sklearn':
                            fout.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (idx, test_labels[idx][0], predict, int(l0), l1, l2, linf))
                        else:
                            fout.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\n' % (idx, test_labels[idx][0], predict[0], int(l0), l1, l2, linf))

                print("\n======== Point {} ({}/{}) finished =========".format(idx, n+1, num_attacks))


                #fout.write('%s\t%s\t%s\t%s\t-%s\t+%s\n' % (idx, int(l0), l2, linf, ','.join(deleted), ','.join(inserted)))
        print("model makes {}/{} correct predictions".format(counter, num_attacks))
        print('model and manual predictions inconsistent on adv samples: {}'.format(model_manual_inconsistent))
        print('model and manual predictions inconsistent on clean samples: {}'.format(clean_sample_inconsistent))
        print('milp fails to have optimal solution: {}'.format(milp_non_optimal))
        print('actual guard val used: {}'.format(attack.guard_val))
        fout.write("\nmodel makes {}/{} correct predictions".format(counter, num_attacks))
        if success_count == 0:
            print("No attack succeeded.")
            fout.close()
        else:

            valid_attacks = counter - len(model_manual_inconsistent) - len(clean_sample_inconsistent)

            avg_l0 = avg_l0/success_count
            avg_l1 = avg_l1/success_count
            avg_l2 = avg_l2/success_count
            avg_linf = avg_linf/success_count
            if args['weight'] != 'No weight':
                avg_l1_weighted = avg_l1_weighted/success_count
            if args['weight'] != 'No weight':
                print('\naverage weighted l1 distortion: {}'.format(avg_l1_weighted))
            fout.write("\nattacked %d/%d points, average l0 distortion:%f, average l1 distortion:%f,  average l2 distortion:%f, average linf distortion:%f"%(counter, num_attacks, avg_l0, avg_l1, avg_l2, avg_linf))
            if args['weight'] != 'No weight':
                fout.write('\naverage weighted l1 distortion: {}'.format(avg_l1_weighted))
            avg_time = (time.time()-global_start) / counter
            fout.write("\navg time:%f"%(avg_time))
            print("\nattacked %d/%d points, average l0 distortion:%f, average l1 distortion:%f, average l2 distortion:%f, average linf distortion:%f"%(counter, num_attacks, avg_l0, avg_l1, avg_l2, avg_linf))
            print("success rate: {}/{} = {}".format(success_count, valid_attacks, success_count * 100.0/valid_attacks))
            print("avg time:%f"%(avg_time))
            print("results saved in", args['out'])
            print("adv examples saved in", args['adv'])
            pickle.dump(all_adv, open(args['adv'], 'wb'))
            fout.write("\nsuccess rate: {}/{}".format(success_count, valid_attacks))
            fout.close()
        return avg_l0, avg_l1, avg_l2, avg_linf, avg_time

if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('-d', '--data', help='test data path')
        parser.add_argument('-m', '--model', help='model path. if it is sklearn model, this is pickled.')
        parser.add_argument('--model_type', type=str, default='xgboost', choices=['xgboost', 'sklearn'], help='choose model type.')
        parser.add_argument('--model_json', type=str, help='the json representation of an sklearn model', required=False)
        parser.add_argument('-c', '--num_classes', type=int, help='number of classes')
        parser.add_argument('-o', '--offset', type=int, default=0, help='start index of attack')
        parser.add_argument('--order', type=int, default=1, choices = [0, 1, 2, -1],  help='order of norm to minimize (-1 for infinity)')
        parser.add_argument('-n', '--num_attacks', type=int, default=3416, help='number of points to be attacked, -1 for attack all')
        parser.add_argument('-g', '--guard_val', type=float, default=GUARD_VAL, help='guard value')
        parser.add_argument('-r', '--round_digits', type=int, default=ROUND_DIGITS, help='number of digits to round threshold')
        parser.add_argument('--round_data', type=int, default=ROUND_DIGITS, help='number of digits to round data')
        parser.add_argument('-w', '--weight', type=str, default="No weight", help='the JSON representation of cost to change each feasture')
        parser.add_argument('--out', help='output csv file name')
        parser.add_argument('--adv', help='sha1 to adv example file pickle')
        parser.add_argument('-t', '--threads', default=8, type=int, help='number of threads')
        parser.add_argument('--feature_start', type=int, default=1, choices=[0,1], help='feature number starts from which index? For cod-rna and higgs, this should be 0.')
        parser.add_argument('--initial_check', default=False, help='whether to check model & manual agree on all test samples', action='store_true')
        parser.add_argument('--no_shuffle', default=False, help='Not shuffle the test data points', action='store_true')
        args = vars(parser.parse_args())
        print(args)
        main(args)