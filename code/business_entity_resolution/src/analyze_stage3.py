"""Inspect completed Stage 3 retrieval without fitting models or changing candidates."""
import argparse
from contextlib import closing,ExitStack
import csv
import json
from pathlib import Path

from blocking_index import connect
from blocking_retrieval import ROUTES
from scoring import parse_ids,f05_counts


def analyze(run_dir,splits_path,index_dir,budgets=(5,10,20,50)):
    run_dir,index_dir=Path(run_dir),Path(index_dir)
    config=json.loads((run_dir/'config.json').read_text(encoding='utf-8'))
    report=json.loads((run_dir/'report.json').read_text(encoding='utf-8'))
    budgets=[k for k in budgets if k<=min(config['retrieval']['budgets'].values())]
    results={k:dict(entities=0,non_singletons=0,recall_sum=0.,oracle_sum=0.,pairs=0) for k in budgets}
    source_recall={source:[0.,0] for source in (2,3)}
    with ExitStack() as stack:
        db=stack.enter_context(closing(connect(run_dir/'candidates.sqlite',True)))
        truth_db=stack.enter_context(closing(connect(splits_path,True)))
        pools={i:stack.enter_context(closing(connect(index_dir/f'train_source{i}.sqlite',True))) for i in (2,3)}
        handle=stack.enter_context((run_dir/'missed_pairs.tsv').open('w',encoding='utf-8',newline=''))
        writer=csv.writer(handle,delimiter='\t',lineterminator='\n')
        writer.writerow(['source1_entity_id','candidate_entity_id','country','query_name','query_address',
                         'candidate_name','candidate_address'])
        missed=0
        for eid,country,name,address in db.execute('SELECT id,country,name,address FROM queries ORDER BY id'):
            truth=parse_ids(truth_db.execute('SELECT matched_ids FROM entities WHERE id=?',(eid,)).fetchone()[0])
            pairs=[(cid,json.loads(meta)) for cid,meta in db.execute(
                'SELECT candidate_id,metadata FROM candidates WHERE s1_id=?',(eid,))]
            candidates={cid for cid,_ in pairs}
            for source in (2,3):
                positives={cid for cid in truth if cid.startswith(f'S{source}-')}
                if positives:
                    source_recall[source][0]+=len(positives & candidates)/len(positives)
                    source_recall[source][1]+=1
            for k,g in results.items():
                selected={cid for cid,meta in pairs if any(v['rank']<=k for v in meta.values())}
                hit=len(selected & truth)
                g['entities']+=1
                g['non_singletons']+=bool(truth)
                g['recall_sum']+=hit/len(truth) if truth else 0
                g['oracle_sum']+=f05_counts(hit,0,len(truth)-hit)
                g['pairs']+=len(selected)
            for cid in sorted(truth-candidates):
                source=int(cid[1])
                other=pools[source].execute('SELECT name,address FROM records WHERE entity_id=?',(cid,)).fetchone()
                if other is None:
                    raise AssertionError(f'True match missing from complete candidate pool: {cid}')
                writer.writerow([eid,cid,country,name,address,*other])
                missed+=1
    for g in results.values():
        g['macro_recall']=g.pop('recall_sum')/g['non_singletons'] if g['non_singletons'] else None
        g['oracle_macro_f05']=g.pop('oracle_sum')/g['entities']
        g['mean_candidates']=g['pairs']/g['entities']
    summary=dict(scope=report['scope'],equal_budget_per_route_per_source=results,missed_true_pairs=missed,
                 source_macro_recall={str(k):dict(recall=v[0]/v[1] if v[1] else None,entities_with_source_matches=v[1])
                                      for k,v in source_recall.items()},
                 note='Budget comparisons reuse identical shortlists and rank metadata; no classifier is fitted.')
    (run_dir/'diagnostics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--splits-db',type=Path,default=Path('reports/stage1/splits.sqlite'))
    p.add_argument('--index-dir',type=Path,default=Path('artifacts/stage3/indexes'))
    a=p.parse_args()
    print(json.dumps(analyze(a.run_dir,a.splits_db,a.index_dir),indent=2))
