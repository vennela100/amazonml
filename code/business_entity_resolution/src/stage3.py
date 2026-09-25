"""Stage 3: resumable retrieval and population-explicit candidate recall."""
import argparse
from collections import defaultdict
from contextlib import closing, contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import time

from blocking_index import build_index, connect, fit_idf, get_state, load_idf, log, set_state, source_signature, prepare_statistics
from blocking_retrieval import Retriever, ROUTES
from scoring import parse_ids, f05_counts
from splits import selection


@contextmanager
def run_lock(path):
    with Path(path).open('a+b') as lock:
        lock.seek(0)
        if not lock.read(1):
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f'Another process holds {path}') from exc
        yield


def prepare_queries(db, query_index, mode, role, limit):
    db.executescript('''
        CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS queries(id TEXT PRIMARY KEY,country TEXT NOT NULL,
            name TEXT,address TEXT,houses TEXT,postals TEXT,done2 INTEGER DEFAULT 0,done3 INTEGER DEFAULT 0);
        CREATE TABLE IF NOT EXISTS candidates(s1_id TEXT,candidate_id TEXT,source INTEGER,
            routes TEXT,best_score REAL,metadata TEXT,PRIMARY KEY(s1_id,candidate_id)) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS pending2 ON queries(id) WHERE done2=0;
        CREATE INDEX IF NOT EXISTS pending3 ON queries(id) WHERE done3=0;
    ''')
    if get_state(db,'queries_ready',False):
        return
    predicate, params = selection(mode,role,alias='')
    with closing(connect(query_index,True)) as src:
        total = src.execute('SELECT count(*) FROM records WHERE '+predicate,params).fetchone()[0]
        sql = ('SELECT entity_id,country,name,address,houses,postals FROM records WHERE '+predicate
               + (' ORDER BY split_hash,entity_id LIMIT ?' if limit else ' ORDER BY rid'))
        with db:
            db.executemany('INSERT INTO queries(id,country,name,address,houses,postals) VALUES (?,?,?,?,?,?)',
                           src.execute(sql,params+(limit,) if limit else params))
            set_state(db,'population_entities',total)
            set_state(db,'queries_ready',True)


def run_source(db,pool,idf,config,source):
    column = 'done2' if source==2 else 'done3'
    retriever = Retriever(pool,idf,config)
    started = time.monotonic()
    previous_seconds = get_state(db,f'source{source}_seconds',0.)
    completed = db.execute(f'SELECT count(*) FROM queries WHERE {column}=1').fetchone()[0]
    total = db.execute('SELECT count(*) FROM queries').fetchone()[0]
    try:
        while True:
            batch = db.execute(f'SELECT id,name,address,houses,postals FROM queries WHERE {column}=0 LIMIT 25').fetchall()
            if not batch:
                break
            with db:
                for eid,name,address,houses,postals in batch:
                    union = retriever.retrieve(dict(name=name,address=address,houses=houses,postals=postals))
                    rows = []
                    for cid,metadata in union.items():
                        if not cid.startswith(f'S{source}-'):
                            raise ValueError(f'Unexpected candidate source: {cid}')
                        rows.append((eid,cid,source,','.join(sorted(metadata)),max(v['score'] for v in metadata.values()),
                                     json.dumps(metadata,sort_keys=True)))
                    db.executemany('INSERT INTO candidates VALUES (?,?,?,?,?,?)',rows)
                    db.execute(f'UPDATE queries SET {column}=1 WHERE id=?',(eid,))
                set_state(db,f'source{source}_seconds',previous_seconds+time.monotonic()-started)
            completed += len(batch)
            log(f'S{source}: {completed:,}/{total:,} queries committed; elapsed {time.monotonic()-started:.1f}s')
    finally:
        retriever.close()


def evaluate(db,splits_path,output,pool_rows):
    if db.execute('SELECT count(*) FROM queries WHERE done2=0 OR done3=0').fetchone()[0]:
        raise ValueError('Cannot evaluate unfinished queries')
    output = Path(output)
    output.mkdir(parents=True,exist_ok=True)
    groups = defaultdict(lambda:dict(entities=0,non_singletons=0,singletons=0,recall_sum=0.,oracle_sum=0.,
                                    true_matches=0,found_matches=0,candidates=0,fully_recalled=0,zero_recalled=0))
    route_hits = {route:[0.,0] for route in ROUTES}
    with closing(connect(splits_path,True)) as truth_db, \
            (output/'per_entity_recall.tsv').open('w',encoding='utf-8',newline='') as handle, \
            (output/'candidate_pairs.tsv').open('w',encoding='utf-8',newline='') as candidate_handle:
        writer = csv.writer(handle,delimiter='\t',lineterminator='\n')
        writer.writerow(['source1_entity_id','country','true_matches','found_matches','candidates','recall','oracle_f05'])
        cw = csv.writer(candidate_handle,delimiter='\t',lineterminator='\n')
        cw.writerow(['source1_entity_id','candidate_entity_ids'])
        for eid,country in db.execute('SELECT id,country FROM queries ORDER BY id'):
            row = truth_db.execute('SELECT matched_ids FROM entities WHERE id=?',(eid,)).fetchone()
            if row is None:
                raise ValueError(f'Missing truth for {eid}')
            truth = parse_ids(row[0])
            candidates = db.execute('SELECT candidate_id,routes FROM candidates WHERE s1_id=? ORDER BY candidate_id',(eid,)).fetchall()
            predicted = {cid for cid,_ in candidates}
            found = len(truth & predicted)
            recall = found/len(truth) if truth else None
            oracle = f05_counts(found,0,len(truth)-found)
            writer.writerow([eid,country,len(truth),found,len(predicted),'' if recall is None else recall,oracle])
            cw.writerow([eid,','.join(sorted(predicted))])
            for key in ('overall','country:'+country):
                g = groups[key]
                g['entities'] += 1
                g['non_singletons'] += bool(truth)
                g['singletons'] += not truth
                g['recall_sum'] += recall or 0
                g['oracle_sum'] += oracle
                g['true_matches'] += len(truth)
                g['found_matches'] += found
                g['candidates'] += len(predicted)
                g['fully_recalled'] += bool(truth) and found==len(truth)
                g['zero_recalled'] += bool(truth) and found==0
            if truth:
                for route in ROUTES:
                    hits = sum(cid in truth and route in routes.split(',') for cid,routes in candidates)
                    route_hits[route][0] += hits/len(truth)
                    route_hits[route][1] += 1
    for g in groups.values():
        g['macro_candidate_recall_non_singletons'] = g.pop('recall_sum')/g['non_singletons'] if g['non_singletons'] else None
        g['oracle_macro_f05_ceiling'] = g.pop('oracle_sum')/g['entities']
        g['all_empty_macro_f05'] = g['singletons']/g['entities']
        g['micro_candidate_recall'] = g['found_matches']/g['true_matches'] if g['true_matches'] else None
        g['mean_candidates'] = g['candidates']/g['entities']
        g['reduction_ratio'] = 1-g['candidates']/(g['entities']*sum(pool_rows.values()))
    if not groups:
        raise ValueError('Empty evaluation population')
    report = dict(status='PASS',scope='full' if groups['overall']['entities']==get_state(db,'population_entities') else 'pilot',
                  population_entities=get_state(db,'population_entities'),pool_rows=pool_rows,groups=dict(groups),
                  retrieval_seconds={str(i):get_state(db,f'source{i}_seconds',0.) for i in (2,3)},
                  route_macro_recall={k:v[0]/v[1] if v[1] else None for k,v in route_hits.items()},
                  notes=['Singleton recall is undefined and excluded from macro candidate recall.',
                         'Oracle F0.5 assumes perfect classification of candidates; it is not model performance.',
                         'Approximate inverted-index shortlists reranked with train-fitted TF-IDF cosine.',
                         'candidate_pairs.tsv is a training retrieval artifact, not a test submission.'])
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name,default in [('normalized-dir','artifacts/stage2/train'),('stage2-report','reports/stage2/report.json'),
                         ('splits-db','reports/stage1/splits.sqlite'),('manifest','reports/stage1/manifest.json'),
                         ('index-dir','artifacts/stage3/indexes'),('output-dir','reports/stage3/grouped_pilot')]:
        p.add_argument('--'+name,type=Path,default=Path(default))
    p.add_argument('--mode',default='grouped')
    p.add_argument('--role',choices=['train','validation'],default='validation')
    p.add_argument('--limit',type=int,default=1000,help='Deterministic pilot count; zero selects entire split')
    p.add_argument('--index-only',action='store_true')
    p.add_argument('--source',type=int,choices=[1,2,3],help='Build one index, requires --index-only')
    for name,default in [('anchors',8),('postings-per-anchor',96),('shortlist',96),('rare-max-df',100),('component-max-df',5000)]:
        p.add_argument('--'+name,type=int,default=default)
    for route in ROUTES:
        p.add_argument('--topk-'+route,type=int,default=20)
    a = p.parse_args()
    if a.limit<0 or any(getattr(a,key)<=0 for key in ('anchors','postings_per_anchor','shortlist','rare_max_df','component_max_df')):
        p.error('Positive limits required; --limit alone allows zero')
    if any(getattr(a,'topk_'+route)<0 for route in ROUTES):
        p.error('Nonnegative route budgets required')
    if a.source and not a.index_only:
        p.error('--source requires --index-only')
    stage2 = json.loads(a.stage2_report.read_text(encoding='utf-8'))
    if stage2['status']!='PASS':
        raise ValueError('Stage 2 must pass first')
    mode = json.loads(a.manifest.read_text(encoding='utf-8'))['modes'][a.mode]
    a.index_dir.mkdir(parents=True,exist_ok=True)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    indexes = {i:a.index_dir/f'train_source{i}.sqlite' for i in (1,2,3)}
    with run_lock(a.output_dir/'run.lock'):
        for source in ([a.source] if a.source else (1,2,3)):
            name = f'train_source{source}.tsv'
            path = a.normalized_dir/name
            if path.stat().st_size!=stage2['files'][name]['output_bytes']:
                raise ValueError(f'Artifact size changed: {path}')
            with run_lock(a.index_dir/f'source{source}.lock'):
                build_index(path,indexes[source],stage2['files'][name]['rows'],a.splits_db if source==1 else None)
                if source in (2,3):
                    prepare_statistics(indexes[source])
        if a.index_only:
            return
        with run_lock(a.index_dir/f'idf_{a.mode}.lock'):
            idf_path = fit_idf(indexes[1],a.index_dir/f'idf_{a.mode}.sqlite',mode)
        config = {key:getattr(a,key) for key in ('anchors','postings_per_anchor','shortlist','rare_max_df','component_max_df')}
        config['budgets'] = {route:getattr(a,'topk_'+route) for route in ROUTES}
        run_config = dict(retrieval=config,mode=mode,role=a.role,limit=a.limit,
                          implementation_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                                 for name in ('stage3.py','blocking_index.py','blocking_retrieval.py')},
                          indexes={str(k):source_signature(v) for k,v in indexes.items()},idf=source_signature(idf_path))
        with closing(connect(a.output_dir/'candidates.sqlite')) as db:
            db.execute('CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT NOT NULL)')
            previous = get_state(db,'config')
            if previous is not None and previous!=run_config:
                raise ValueError('Run configuration changed; choose a new output directory')
            set_state(db,'config',run_config)
            db.commit()
            (a.output_dir/'config.json').write_text(json.dumps(run_config,indent=2),encoding='utf-8')
            prepare_queries(db,indexes[1],mode,a.role,a.limit)
            idf = load_idf(idf_path)
            for source in (2,3):
                run_source(db,indexes[source],idf,config,source)
            metrics = evaluate(db,a.splits_db,a.output_dir,{str(i):stage2['files'][f'train_source{i}.tsv']['rows'] for i in (2,3)})
            log(json.dumps(metrics,indent=2))


if __name__=='__main__':
    main()
