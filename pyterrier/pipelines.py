from warnings import warn
from tqdm import tqdm
from collections import defaultdict
import itertools
import os
import pandas as pd
import numpy as np
from .utils import Utils
from .transformer import TransformerBase, EstimatorBase
from .model import add_ranks
import deprecation

def _bold_cols(data, col_type):
    if not data.name in col_type:
        return [''] * len(data)
    
    colormax_attr = f'font-weight: bold'
    colormaxlast_attr = f'font-weight: bold'
    if col_type[data.name] == "+":  
        max_value = data.max()
    else:
        max_value = data.min()
    
    is_max = [colormax_attr if v == max_value else '' for v in data]
    is_max[len(data) - list(reversed(data)).index(max_value) -  1] = colormaxlast_attr
    return is_max

def _color_cols(data, col_type, 
                       colormax='antiquewhite', colormaxlast='lightgreen', 
                       colormin='antiquewhite', colorminlast='lightgreen' ):
    if not data.name in col_type:
      return [''] * len(data)
    
    if col_type[data.name] == "+":
      colormax_attr = f'background-color: {colormax}'
      colormaxlast_attr = f'background-color: {colormaxlast}'
      max_value = data.max()
    else:
      colormax_attr = f'background-color: {colormin}'
      colormaxlast_attr = f'background-color: {colorminlast}'
      max_value = data.min()
    
    is_max = [colormax_attr if v == max_value else '' for v in data]
    is_max[len(data) - list(reversed(data)).index(max_value) -  1] = colormaxlast_attr
    return is_max

# The algorithm for grid search
def GridSearch(pipeline : TransformerBase, topics : pd.DataFrame, qrels : pd.DataFrame, param_map : dict, metric="ndcg", verbose=True):
    """
    Grid searches a set of named parameters on a given pipeline. The topics, qrels must be specified.
    The trec_eval measure name can be optionally specified.

    Transformers must be uniquely identified using the 'id' constructor kwarg. The parameter being
    varied must be changable using the :func:`set_parameter()` method. This means instance variables,
    as well as controls in the case of BatchRetrieve.

    GridSearch currently has a sequential implementation - this is likely to change in the future.

    Args:
        - pipeline(TransformerBase): a transformer or pipeline
        - topics(DataFrame): topics to tune upon
        - qrels(DataFrame): qrels to tune upon
        - param_map(dict): a two-level dictionary, mapping transformer id to param name to a list of values
        - metric(string): name of the metric to tune. Defaults to "ndcg".
        - verbose(bool): whether to display progress bars or not 

    Returns:
        - A tuple containing the best transformer, and a two-level parameter map of the identified best settings

    Raises:
        - KeyError: if no transformer with the specified id can be found
        - ValueError: if a specified transformer does not have such a parameter
    
    """

    #Store the all parameter names and candidate values into a dictionary, keyed by a tuple of transformer id and parameter name
    #such as {('id1', 'wmodel'): ['BM25', 'PL2'], ('id1', 'c'): [0.1, 0.2, 0.3], ('id2', 'lr'): [0.001, 0.01, 0.1]}
    candi_dict = { (tran_id, param_name) : param_map[tran_id][param_name] for tran_id in param_map for param_name in param_map[tran_id]}
    if len(candi_dict) == 0:
        raise ValueError("No parameters specified to optimise")

    for tran_id in param_map:
        if pipeline.get_transformer(tran_id) is None:
            raise KeyError("No such transformer with id %s in retrieval pipeline %s" % (tran_id, str(pipeline)))

    # Iterate the candidate values in different combinations
    items = sorted(candi_dict.items())    
    keys,values = zip(*items)
    combinations = list(itertools.product(*values))
    
    eval_list = []
    #for each combination of parameter values
    for v in tqdm(combinations, total=len(combinations), desc="GridSearch", mininterval=0.3) if verbose else combinations:
        #'params' is every combination of candidates
        params = dict(zip(keys,v))
        parameter_list = []

        #Set the parameter value in the corresponding transformer of the pipeline
        for pipe_id, param_name in params:
            pipeline.get_transformer(pipe_id).set_parameter(param_name,params[pipe_id,param_name])
            # such as ('id1', 'wmodel', 'BM25')
            parameter_list.append((pipe_id,param_name,params[pipe_id,param_name]))
            
        # using topics and evaluation
        res = pipeline.transform(topics)
        eval_score = Utils.evaluate(res, qrels, metrics=[metric], perquery=False)[metric]
        # eval_list has the form [ ([('id1', 'wmodel', 'BM25'),('id1', 'c', 0.2),('id2', 'lr', '0.001')],0.2654)], where 0.2654 is the evaluation score.
        eval_list.append( (parameter_list, eval_score) )


    # identify the best setting
    best_score = 0
    max_index = 0
    for i, (param_list, score) in enumerate(eval_list):
        if score > best_score:
            best_score = score
            max_index = i
    best_params, _ = eval_list[max_index]
    
    best_params_map = { tran_id : {} for tran_id in param_map }
    for tran_id, param_name, param_value in best_params:
        best_params_map[tran_id][param_name] = param_value

    # configure the pipeline
    for i in range(len(best_params)):
        if not hasattr(pipeline,"id"):
            pipeline.get_transformer(best_params[i][0]).set_parameter(best_params[i][1],best_params[i][2])
        else:
            pipeline.set_parameter(best_params[i][1],best_params[i][2])
    best_transformer = pipeline

    # display the best results
    if verbose:
        #print best evaluation results
        print("The best %s score is: %f" % (metric, best_score))
        #print the best param map  
        print("The best parameters map is :")
        for i in range(len(best_params)):
            print(best_params[i])

    return best_transformer, best_params_map
    
# The algorithm for grid search with cross validation
def GridSearchCV(pipeline : TransformerBase, topics : pd.DataFrame, qrels : pd.DataFrame, param_map : dict, metric='ndcg', num_folds=5, **kwargs):
    from sklearn.model_selection import KFold
    import numpy as np
    import pandas as pd
    all_split_scores={}
    all_params=[]

    for train_index, test_index in KFold(n_splits=num_folds).split(topics):
        topics_train, topics_test = topics.iloc[train_index],topics.iloc[test_index]
        best_transformer, params = GridSearch(pipeline, topics_train, qrels, param_map, metric=metric)
        all_params.append(params)

        test_res = best_transformer.transform(topics_test)
        test_eval = Utils.evaluate(test_res, qrels, metrics=[metric], perquery=True)
        all_split_scores.update(test_eval)
    return all_split_scores, all_params

def Experiment(retr_systems, topics, qrels, eval_metrics, names=None, perquery=False, dataframe=True, baseline=None, highlight=None, round=None):
    """
    Allows easy comparison of multiple retrieval transformer pipelines using a common set of topics, and
    identical evaluation measures computed using the same qrels. In essence, each transformer is applied on 
    the provided set of topics. Then the named trec_eval evaluation measures are computed 
    (using `pt.Utils.evaluate()`) for each system.

    Args:
        retr_systems(list): A list of transformers to evaluate. If you already have the results for one 
            (or more) of your systems, a results dataframe can also be used here. Results produced by 
            the transformers must have "qid", "docno", "score", "rank" columns.
        topics: Either a path to a topics file or a pandas.Dataframe with columns=['qid', 'query']
        qrels: Either a path to a qrels file or a pandas.Dataframe with columns=['qid','docno', 'label']   
        eval_metrics(list): Which evaluation metrics to use. E.g. ['map']
        names(list): List of names for each retrieval system when presenting the results.
            Default=None. If None: Obtains the `str()` representation of each transformer as its name.
        perquery(bool): If true return each metric for each query, else return mean metrics across all queries. Default=False.
        dataframe(bool): If True return results as a dataframe. Else as a dictionary of dictionaries. Default=True.
        baseline(int): If set to the index of an item of the retr_system list, will calculate the number of queries 
            improved, degraded and the statistical significance (paired t-test p value) for each measure.
            Default=None: If None, no additional columns added for each measure
        highlight(str): If `highlight="bold"`, highlights in bold the best measure value in each column; 
            if `highlight="color"` or `"colour"`, then the cell with the highest metric value will have a green background.
        round(int): How many decimal places to round each measure value to. This can be a dictionary mapping measure name to number of decimal places.
            Default is None, which is no rounding.

    Returns:
        A Dataframe with each retrieval system with each metric evaluated.
    """
    
    
    # map to the old signature of Experiment
    warn_old_sig=False
    if isinstance(retr_systems, pd.DataFrame) and isinstance(topics, list):
        tmp = topics
        topics = retr_systems
        retr_systems = tmp
        warn_old_sig = True
    if isinstance(eval_metrics, pd.DataFrame) and isinstance(qrels, list):
        tmp = eval_metrics
        eval_metrics = qrels
        qrels = tmp
        warn_old_sig = True
    if warn_old_sig:
        warn("Signature of Experiment() is now (retr_systems, topics, qrels, eval_metrics), please update your code", DeprecationWarning, 2)
    
    if baseline is not None:
        assert int(baseline) >= 0 and int(baseline) < len(retr_systems)
        assert not perquery

    if isinstance(topics, str):
        if os.path.isfile(topics):
            topics = Utils.parse_trec_topics_file(topics)
    if isinstance(qrels, str):
        if os.path.isfile(qrels):
            qrels = Utils.parse_qrels(qrels)
    from timeit import default_timer as timer

    if round is not None:
        if isinstance(round, int):
            assert round >= 0, "round argument should be integer >= 0, not %s" % str(round)
        elif isinstance(round, dict):
            assert not perquery, "Sorry, per-measure rounding only support when reporting means" 
            for k,v in round.items():
                assert isinstance(v, int) and v >= 0, "rounding number for measure %s should be integer >= 0, not %s" % (k, str(v))
        else:
            raise ValueError("Argument round should be an integer or a dictionary")

    def _apply_round(measure, value):
        if round is not None and isinstance(round, int):
            value = builtins.round(value, round)
        if round is not None and isinstance(round, dict) and measure in round:
            value = builtins.round(value, round[measure])
        return value

    results = []
    times=[]
    neednames = names is None
    if neednames:
        names = []
    elif len(names) != len(retr_systems):
        raise ValueError("names should be the same length as retr_systems")
    for system in retr_systems:
        # if its a DataFrame, use it as the results
        if isinstance(system, pd.DataFrame):
            results.append(system)
            times.append(0)
        else:
            starttime = timer()            
            results.append(system.transform(topics))
            endtime = timer()
            times.append( (endtime - starttime) * 1000.)
            
        if neednames:
            names.append(str(system))

    qrels_dict = Utils.convert_qrels_to_dict(qrels)
    all_qids = topics["qid"].values

    evalsRows=[]
    evalDict={}
    evalDictsPerQ=[]
    actual_metric_names=[]
    mrt_needed = False
    if "mrt" in eval_metrics:
        mrt_needed = True
        eval_metrics.remove("mrt")
    for name,res,time in zip(names, results, times):
        evalMeasuresDict = Utils.evaluate(res, qrels_dict, metrics=eval_metrics, perquery=perquery or baseline is not None)
        
        if perquery or baseline is not None:
            # this ensures that all queries are present in various dictionaries
            # its equivalent to "trec_eval -c"
            (evalMeasuresDict, missing) = Utils.ensure(evalMeasuresDict, eval_metrics, all_qids)
            if missing > 0:
                warn("%s was missing %d queries, expected %d" % (name, missing, len(all_qids) ))

        if baseline is not None:
            evalDictsPerQ.append(evalMeasuresDict)
            evalMeasuresDict = Utils.mean_of_measures(evalMeasuresDict)

        if mrt_needed:
            evalMeasuresDict["mrt"] = time / float(len(all_qids))

        if perquery:
            for qid in all_qids:
                for measurename in evalMeasuresDict[qid]:
                    evalsRows.append([
                        name, 
                        qid, 
                        measurename, 
                        _apply_round(
                            measurename, 
                            evalMeasuresDict[qid][measurename]
                        ) 
                    ])
            evalDict[name] = evalMeasuresDict
        else:
            import builtins
            actual_metric_names = list(evalMeasuresDict.keys())
            # gather mean values, applying rounding if necessary
            evalMeasures=[ _apply_round(m, evalMeasuresDict[m]) for m in actual_metric_names]

            evalsRows.append([name]+evalMeasures)
            evalDict[name] = evalMeasures
    if dataframe:
        if perquery:
            df = pd.DataFrame(evalsRows, columns=["name", "qid", "measure", "value"])
            if round is not None:
                df["value"] = df["value"].round(num_dp)
            return df

        highlight_cols = { m : "+"  for m in actual_metric_names }
        if mrt_needed:
            highlight_cols["mrt"] = "-"

        if baseline is not None:
            assert len(evalDictsPerQ) == len(retr_systems)
            from scipy import stats
            baselinePerQuery={}
            per_q_metrics = actual_metric_names.copy()
            if mrt_needed:
                per_q_metrics.remove("mrt")

            for m in per_q_metrics:
                baselinePerQuery[m] = np.array([ evalDictsPerQ[baseline][q][m] for q in evalDictsPerQ[baseline] ])

            for i in range(0, len(retr_systems)):
                additionals=[]
                if i == baseline:
                    additionals = [None] * (3*len(per_q_metrics))
                else:
                    for m in per_q_metrics:
                        # we iterate through queries based on the baseline, in case run has different order
                        perQuery = np.array( [ evalDictsPerQ[i][q][m] for q in evalDictsPerQ[baseline] ])
                        delta_plus = (perQuery > baselinePerQuery[m]).sum()
                        delta_minus = (perQuery < baselinePerQuery[m]).sum()
                        p = stats.ttest_rel(perQuery, baselinePerQuery[m])[1]
                        additionals.extend([delta_plus, delta_minus, p])
                evalsRows[i].extend(additionals)
            delta_names=[]
            for m in per_q_metrics:
                delta_names.append("%s +" % m)
                highlight_cols["%s +" % m] = "+"
                delta_names.append("%s -" % m)
                highlight_cols["%s -" % m] = "-"
                delta_names.append("%s p-value" % m)
            actual_metric_names.extend(delta_names)

        df = pd.DataFrame(evalsRows, columns=["name"] + actual_metric_names)
        
        if highlight == "color" or highlight == "colour" :
            df = df.style.apply(_color_cols, axis=0, col_type=highlight_cols)
        elif highlight == "bold":
            df = df.style.apply(_bold_cols, axis=0, col_type=highlight_cols)
            
        return df 
    return evalDict

from .ltr import RegressionTransformer, LTRTransformer
@deprecation.deprecated(deprecated_in="0.3.0",
                        details="Please use pt.ltr.apply_learned_model(learner, form='regression')")
class LTR_pipeline(RegressionTransformer):
    pass

@deprecation.deprecated(deprecated_in="0.3.0",
                        details="Please use pt.ltr.apply_learned_model(learner, form='ltr')")
class XGBoostLTR_pipeline(LTRTransformer):
    pass


class PerQueryMaxMinScoreTransformer(TransformerBase):
    '''
    applies per-query maxmin scaling on the input scores
    '''
    
    def transform(self, topics_and_res):
        from sklearn.preprocessing import minmax_scale
        topics_and_res = topics_and_res.copy()
        topics_and_res["score"] = topics_and_res.groupby('qid')["score"].transform(lambda x: minmax_scale(x))
        return topics_and_res
