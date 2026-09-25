"""Independently re-read Stage 3 exports and verify coverage, IDs, and metric parity."""
import argparse
from contextlib import closing,ExitStack
import json
import math
from pathlib import Path

from blocking_index import connect,get_state
from scoring import MacroMetrics,parse_ids,read_tsv,validate_s1


def validate(run_dir,splits_path,index_dir):
    run_dir,index_dir=Path(run_dir),Path(index_dir)
    report=json.loads((run_dir/'report.json').read_text(encoding='utf-8'))
    oracle=MacroMetrics()
    recalls=[]
    pair_count=0
    with ExitStack() as stack:
        db=stack.enter_context(closing(connect(run_dir/'candidates.sqlite',True)))
        labels=stack.enter_context(closing(connect(splits_path,True)))
        pools={i:stack.enter_context(closing(connect(index_dir/f'train_source{i}.sqlite',True))) for i in (2,3)}
        expected={eid for eid, in db.execute('SELECT id FROM queries')}
        seen=set()
        for eid,raw in read_tsv(run_dir/'candidate_pairs.tsv',['source1_entity_id','candidate_entity_ids']):
            validate_s1(eid)
            if eid not in expected or eid in seen:
                raise ValueError(f'Unexpected or repeated S1: {eid}')
            seen.add(eid)
            candidates=parse_ids(raw)
            stored={cid for cid, in db.execute('SELECT candidate_id FROM candidates WHERE s1_id=?',(eid,))}
            if candidates!=stored:
                raise ValueError(f'Export differs from stored candidate union: {eid}')
            for source,pool in pools.items():
                ids=sorted(cid for cid in candidates if cid.startswith(f'S{source}-'))
                for offset in range(0,len(ids),500):
                    batch=ids[offset:offset+500]
                    count=pool.execute('SELECT count(*) FROM records WHERE entity_id IN ('+
                                       ','.join('?' for _ in batch)+')',batch).fetchone()[0]
                    if count!=len(batch):
                        raise ValueError(f'Unknown candidate ID for {eid}')
            truth=parse_ids(labels.execute('SELECT matched_ids FROM entities WHERE id=?',(eid,)).fetchone()[0])
            oracle.add(truth,truth & candidates)
            if truth:
                recalls.append(len(truth & candidates)/len(truth))
            pair_count+=len(candidates)
        if seen!=expected:
            raise ValueError('Missing S1 rows in export')
        if db.execute('SELECT count(*) FROM queries WHERE done2=0 OR done3=0').fetchone()[0]:
            raise ValueError('Incomplete retrieval checkpoints')
    macro_recall=math.fsum(recalls)/len(recalls) if recalls else None
    macro_oracle=oracle.result()['macro_f05']
    g=report['groups']['overall']
    if pair_count!=g['candidates'] or len(seen)!=g['entities']:
        raise ValueError('Report coverage/count mismatch')
    if macro_recall is None:
        if g['macro_candidate_recall_non_singletons'] is not None:
            raise ValueError('Undefined recall mismatch')
    elif not math.isclose(macro_recall,g['macro_candidate_recall_non_singletons'],abs_tol=1e-12,rel_tol=0):
        raise ValueError('Macro recall mismatch')
    if not math.isclose(macro_oracle,g['oracle_macro_f05_ceiling'],abs_tol=1e-12,rel_tol=0):
        raise ValueError('Oracle F0.5 differs from Stage 1 scorer')
    result=dict(status='PASS',scope=report['scope'],entities=len(seen),pairs=pair_count,
                macro_candidate_recall=macro_recall,oracle_macro_f05=macro_oracle,
                checks=['exact query coverage including empties','unique valid S2/S3 IDs',
                        'candidate existence in complete training pools','export equals stored union',
                        'both sources completed for every query','independent metric parity with Stage 1 scorer'])
    (run_dir/'validation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--splits-db',type=Path,default=Path('reports/stage1/splits.sqlite'))
    p.add_argument('--index-dir',type=Path,default=Path('artifacts/stage3/indexes'))
    a=p.parse_args()
    print(json.dumps(validate(a.run_dir,a.splits_db,a.index_dir),indent=2))
