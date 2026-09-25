import csv
import json
import math
from pathlib import Path
import sqlite3
import shutil
import tempfile
import unittest
from unittest.mock import patch

from blocking_index import build_index,connect,fit_idf,load_idf,source_signature,get_state,set_state,prepare_statistics
from blocking_retrieval import Retriever,ROUTES,cosine,grams,vector
from normalization import OUTPUT_COLUMNS
from stage3 import prepare_queries,run_source,evaluate
from analyze_stage3 import analyze
from validate_stage3 import validate
from scoring import read_tsv


class BlockingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.splits = self.root/'splits.sqlite'
        with sqlite3.connect(self.splits) as db:
            db.execute('CREATE TABLE entities(id TEXT PRIMARY KEY,country TEXT,matched_ids TEXT,grouped_role TEXT,split_hash TEXT)')
            db.executemany('INSERT INTO entities VALUES (?,?,?,?,?)',[
                ('S1-t','US','','train','0'),('S1-v','France','S2-a,S3-a','validation','1'),
                ('S1-empty','France','','validation','2'),('S1-missed','France','S2-z','validation','3')])
        self.q = self.make_index('source1',[
            ('S1-t','common cafe','10 main street','US',['10'],[]),
            ('S1-v','uniquerare café','20 rue paris','France',['20'],['75001']),
            ('S1-empty','','','France',[],[]),('S1-missed','absentword','','France',[],[])],self.splits)
        self.pool = self.make_index('source2',[
            ('S2-a','uniquerare café','20 rue paris','France',['20'],['75001']),
            ('S2-b','common cafe','10 main street','US',['10'],[]),
            ('S2-c','हिन्दी दुकान','30 बाजार','India',['30'],[])])
        self.idf_path = fit_idf(self.q,self.root/'idf.sqlite',{'type':'grouped'})
        self.idf = load_idf(self.idf_path)
        self.config = dict(anchors=8,postings_per_anchor=50,shortlist=50,rare_max_df=100,
                           component_max_df=5000,budgets={route:2 for route in ROUTES})

    def tearDown(self):
        self.tmp.cleanup()

    def make_index(self,name,records,splits=None):
        path = self.root/(name+'.tsv')
        with path.open('w',encoding='utf-8',newline='') as handle:
            writer = csv.writer(handle,delimiter='\t',lineterminator='\n')
            writer.writerow(OUTPUT_COLUMNS)
            for eid,n,a,c,h,p in records:
                row = ['']*len(OUTPUT_COLUMNS)
                row[0:4] = [eid,n,a,c]
                row[5:7] = [n,a]
                row[8],row[12] = json.dumps(h),json.dumps(p)
                writer.writerow(row)
        index = self.root/(name+'.sqlite')
        build_index(path,index,len(records),splits,batch_size=2)
        return index

    def test_idf_excludes_validation(self):
        self.assertNotIn('uni',self.idf['name'])
        self.assertAlmostEqual(self.idf['name']['com'],1.)
        self.assertAlmostEqual(self.idf['unseen'],math.log(2)+1)

    def test_country_transfer_refits_training_country_only(self):
        path=fit_idf(self.q,self.root/'transfer.sqlite',
                     {'type':'country_transfer','train_country':'France','validation_country':'US'})
        weights=load_idf(path)
        self.assertIn('uni',weights['name'])
        self.assertNotIn('com',weights['name'])
        self.assertAlmostEqual(weights['unseen'],math.log(4)+1)

    def test_real_weighted_cosine(self):
        a=vector({'aaa':2,'bbb':1},{'aaa':2,'bbb':3},1)
        b=vector({'aaa':1},{'aaa':2},1)
        self.assertAlmostEqual(cosine(a,b),.8)
        self.assertEqual(grams(''),{})

    def test_all_routes_df_one_and_open_country(self):
        prepare_statistics(self.pool)
        with sqlite3.connect(self.pool) as db:
            self.assertEqual(db.execute("SELECT df FROM statistics WHERE kind='word' AND term='uniquerare' AND field='name'").fetchone()[0],1)
        r=Retriever(self.pool,self.idf,self.config)
        try:
            result=r.retrieve(dict(name='uniquerare café',address='20 rue paris',houses='["20"]',postals='["75001"]'))
            self.assertEqual(set(result['S2-a']),set(ROUTES))
            self.assertEqual(r.df('word','name','uniquerare'),1)
            for metadata in result['S2-a'].values():
                self.assertEqual(metadata['rank'],1)
                self.assertAlmostEqual(metadata['score'],1.)
        finally:
            r.close()

    def test_missing_and_unicode(self):
        r=Retriever(self.pool,self.idf,self.config)
        try:
            self.assertEqual(r.retrieve(dict(name='',address='',houses='[]',postals='[]')), {})
            result=r.retrieve(dict(name='हिन्दी दुकान',address='',houses='[]',postals='[]'))
            self.assertIn('rare',result['S2-c'])
        finally:
            r.close()

    def test_budget_and_determinism(self):
        self.config['budgets']={route:1 for route in ROUTES}
        r=Retriever(self.pool,self.idf,self.config)
        try:
            q=dict(name='common cafe',address='10 main street',houses='["10"]',postals='[]')
            a=r.retrieve(q)
            self.assertEqual(a,r.retrieve(q))
            for route in ROUTES:
                self.assertLessEqual(sum(route in m for m in a.values()),1)
        finally:
            r.close()

    def test_completed_index_reuse_is_immutable(self):
        before=source_signature(self.q)
        build_index(self.root/'source1.tsv',self.q,4,self.splits)
        self.assertEqual(before,source_signature(self.q))

    def test_interrupted_index_resumes_committed_batches(self):
        target=self.root/'interrupted.sqlite'
        source=self.root/'source2.tsv'
        def interrupted(path,header):
            for i,row in enumerate(read_tsv(path,header)):
                if i==2:
                    raise RuntimeError('simulated interruption')
                yield row
        with patch('blocking_index.read_tsv',interrupted):
            with self.assertRaisesRegex(RuntimeError,'simulated'):
                build_index(source,target,3,batch_size=2)
        with sqlite3.connect(target) as db:
            self.assertEqual(get_state(db,'rows'),2)
            self.assertFalse(get_state(db,'complete',False))
        build_index(source,target,3,batch_size=2)
        with sqlite3.connect(target) as db:
            self.assertEqual(db.execute('SELECT count(*) FROM records').fetchone()[0],3)
            self.assertTrue(get_state(db,'complete'))

    def test_idf_finalization_resumes(self):
        with sqlite3.connect(self.idf_path) as db:
            set_state(db,'complete',False)
            db.execute('DELETE FROM idf')
        fit_idf(self.q,self.idf_path,{'type':'grouped'})
        self.assertEqual(self.idf,load_idf(self.idf_path))

    def test_macro_recall_weights_entities_not_pairs(self):
        db=connect(self.root/'manual.sqlite')
        try:
            prepare_queries(db,self.q,{'type':'grouped'},'validation',0)
            db.execute('UPDATE queries SET done2=1,done3=1')
            db.executemany('INSERT INTO candidates VALUES (?,?,?,?,?,?)',[
                ('S1-v','S2-a',2,'name',.9,'{}'),
                ('S1-missed','S2-z',2,'address',.9,'{}'),
                ('S1-empty','S2-b',2,'name',.9,'{}')])
            db.commit()
            g=evaluate(db,self.splits,self.root/'manual',{'2':3,'3':1})['groups']['overall']
            self.assertAlmostEqual(g['macro_candidate_recall_non_singletons'],.75)
            self.assertAlmostEqual(g['micro_candidate_recall'],2/3)
            self.assertAlmostEqual(g['oracle_macro_f05_ceiling'],(5/6+1+1)/3)
            self.assertEqual(g['zero_recalled'],0)
        finally:
            db.close()

    def test_scope_empty_rows_oracle_and_resume(self):
        pool3=self.make_index('source3',[('S3-a','uniquerare café','20 rue paris','France',['20'],['75001'])])
        db=connect(self.root/'run.sqlite')
        try:
            prepare_queries(db,self.q,{'type':'grouped'},'validation',2)
            run_source(db,self.pool,self.idf,self.config,2)
            count=db.execute('SELECT count(*) FROM candidates').fetchone()[0]
            run_source(db,self.pool,self.idf,self.config,2)
            self.assertEqual(count,db.execute('SELECT count(*) FROM candidates').fetchone()[0])
            with self.assertRaises(ValueError):
                evaluate(db,self.splits,self.root/'out',{'2':3,'3':1})
            run_source(db,pool3,self.idf,self.config,3)
            report=evaluate(db,self.splits,self.root/'out',{'2':3,'3':1})
            self.assertEqual(report['scope'],'pilot')
            g=report['groups']['overall']
            self.assertEqual(g['entities'],2)
            self.assertEqual(g['macro_candidate_recall_non_singletons'],1.)
            self.assertEqual(g['oracle_macro_f05_ceiling'],1.)
            self.assertEqual(g['all_empty_macro_f05'],.5)
            lines=(self.root/'out/candidate_pairs.tsv').read_text().splitlines()
            self.assertIn('S1-empty\t',lines)
            (self.root/'out/config.json').write_text(json.dumps({'retrieval':self.config}))
            shutil.copyfile(self.pool,self.root/'train_source2.sqlite')
            shutil.copyfile(pool3,self.root/'train_source3.sqlite')
            with sqlite3.connect(self.root/'out/candidates.sqlite') as snapshot:
                db.backup(snapshot)
            diagnostics=analyze(self.root/'out',self.splits,self.root,budgets=(1,2))
            self.assertEqual(diagnostics['missed_true_pairs'],0)
            self.assertEqual(diagnostics['equal_budget_per_route_per_source'][1]['macro_recall'],1.)
            self.assertEqual(diagnostics['source_macro_recall']['3']['recall'],1.)
            checked=validate(self.root/'out',self.splits,self.root)
            self.assertEqual(checked['status'],'PASS')
            self.assertEqual(checked['oracle_macro_f05'],1.)
            with (self.root/'out/candidate_pairs.tsv').open('a',encoding='utf-8') as bad_export:
                bad_export.write('S1-empty\t\n')
            with self.assertRaisesRegex(ValueError,'repeated S1'):
                validate(self.root/'out',self.splits,self.root)
        finally:
            db.close()


if __name__=='__main__':
    unittest.main()
