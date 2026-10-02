"""Disk-backed text indexes and train-only character-trigram IDF statistics."""
from __future__ import annotations

from contextlib import closing
import csv
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import time

from normalization import OUTPUT_COLUMNS
from scoring import read_tsv
from splits import selection

INDEX_VERSION = 1


def log(message):
    print(message, flush=True)


def connect(path, readonly=False):
    if readonly:
        db = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=60)
    else:
        db = sqlite3.connect(path, timeout=60)
    db.execute('PRAGMA cache_size=-65536')
    db.execute('PRAGMA temp_store=FILE')
    return db


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_signature(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}


def schema(db):
    db.executescript('''
        CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS records(
            rid INTEGER PRIMARY KEY, entity_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL, address TEXT NOT NULL, country TEXT NOT NULL,
            houses TEXT NOT NULL, postals TEXT NOT NULL,
            grouped_role TEXT NOT NULL, split_hash TEXT NOT NULL);
        CREATE VIRTUAL TABLE IF NOT EXISTS grams USING fts5(
            name,address,content='records',content_rowid='rid',
            tokenize='trigram',detail=column);
        CREATE VIRTUAL TABLE IF NOT EXISTS words USING fts5(
            name,address,content='records',content_rowid='rid',
            tokenize='ascii',detail=column);
        CREATE VIRTUAL TABLE IF NOT EXISTS gram_vocab USING fts5vocab(grams,'col');
        CREATE VIRTUAL TABLE IF NOT EXISTS word_vocab USING fts5vocab(words,'col');
        CREATE TABLE IF NOT EXISTS components(
            kind TEXT, value TEXT, rid INTEGER,
            PRIMARY KEY(kind,value,rid)) WITHOUT ROWID;
    ''')


def set_state(db, key, value):
    db.execute('INSERT OR REPLACE INTO state VALUES (?,?)', (key, json.dumps(value)))


def get_state(db, key, default=None):
    row = db.execute('SELECT value FROM state WHERE key=?', (key,)).fetchone()
    return json.loads(row[0]) if row else default


def build_index(source_path, index_path, expected_rows, splits_path=None, batch_size=10000):
    """Atomic batch checkpoints; repeated invocation skips already committed input rows."""
    started = time.monotonic()
    source_path, index_path = Path(source_path), Path(index_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    config = {'version': INDEX_VERSION, 'source': source_signature(source_path),
              'expected_rows': expected_rows,
              'splits': source_signature(splits_path) if splits_path else None}
    with closing(connect(index_path)) as db:
        schema(db)
        previous = get_state(db, 'config')
        if previous is not None and previous != config:
            raise ValueError(f'{index_path}: input/configuration changed; use a fresh index')
        if get_state(db, 'complete', False):
            log(f'Index ready: {index_path.name} ({expected_rows:,} records)')
            return index_path
        set_state(db, 'config', config)
        db.commit()
        committed = get_state(db, 'rows', 0)
        roles = {}
        if splits_path:
            with closing(connect(splits_path, True)) as labels:
                roles = {eid: (role, split_hash) for eid, role, split_hash in labels.execute(
                    'SELECT id,grouped_role,split_hash FROM entities')}
        records, texts, components = [], [], []

        def flush(number):
            with db:
                db.executemany('INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?)', records)
                db.executemany('INSERT INTO grams(rowid,name,address) VALUES (?,?,?)', texts)
                db.executemany('INSERT INTO words(rowid,name,address) VALUES (?,?,?)', texts)
                db.executemany('INSERT OR IGNORE INTO components VALUES (?,?,?)', components)
                set_state(db, 'rows', number)
            records.clear()
            texts.clear()
            components.clear()

        number = 0
        for number, row in enumerate(read_tsv(source_path, OUTPUT_COLUMNS), 1):
            if number <= committed:
                continue
            eid, country, name, address = row[0], row[3], row[5], row[6]
            houses, postals = row[8], row[12]
            role, split_hash = roles[eid] if splits_path else ('', '')
            records.append((number, eid, name, address, country, houses, postals, role, split_hash))
            texts.append((number, name, address))
            for kind, raw in (('house', houses), ('postal', postals)):
                values = json.loads(raw)
                if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                    raise ValueError(f'Invalid {kind} list: {eid}')
                components.extend((kind, value, number) for value in set(values))
            if len(records) >= batch_size:
                flush(number)
                if number % 100000 == 0:
                    log(f'{index_path.name}: indexed {number:,}/{expected_rows:,}')
        if records:
            flush(number)
        roles.clear()
        if number != expected_rows:
            raise ValueError(f'{source_path}: expected {expected_rows:,} rows, found {number:,}')
        if source_signature(source_path) != config['source']:
            raise ValueError('Source changed during indexing')
        # Optimize after the complete corpus is present. Failure leaves resumable records.
        log(f'{index_path.name}: compacting inverted indexes...')
        db.execute("INSERT INTO grams(grams) VALUES('optimize')")
        db.execute("INSERT INTO words(words) VALUES('optimize')")
        set_state(db, 'complete', True)
        set_state(db, 'elapsed_seconds', time.monotonic() - started)
        db.commit()
    log(f'Index complete: {index_path.name}, {number:,} records')
    return index_path


def fit_idf(query_index, output_path, mode):
    """Native trigram document frequencies, fitted ONLY to the selected training S1s."""
    output_path = Path(output_path)
    with closing(connect(query_index, True)) as queries:
        if not get_state(queries, 'complete', False):
            raise ValueError('Complete the S1 index before fitting IDF')
    config = {'version': INDEX_VERSION, 'query_index': source_signature(query_index), 'mode': mode}
    if output_path.exists():
        with closing(connect(output_path, True)) as db:
            if get_state(db, 'config') != config:
                raise ValueError(f'Changed IDF input/configuration: {output_path}')
            if get_state(db, 'complete', False):
                return output_path
    predicate, parameters = selection(mode, 'train', alias='')
    with closing(connect(query_index, True)) as queries, closing(connect(output_path)) as db:
        db.executescript('''
            CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS fit USING fts5(name,address,tokenize='trigram',detail=column);
            CREATE VIRTUAL TABLE IF NOT EXISTS col_vocab USING fts5vocab(fit,'col');
            CREATE VIRTUAL TABLE IF NOT EXISTS union_vocab USING fts5vocab(fit,'row');
            CREATE TABLE IF NOT EXISTS idf(route TEXT,gram TEXT,value REAL,PRIMARY KEY(route,gram)) WITHOUT ROWID;
        ''')
        set_state(db, 'config', config)
        db.commit()
        last_rid = get_state(db, 'last_rid', 0)
        cursor = queries.execute('SELECT rid,name,address FROM records WHERE ' + predicate + ' AND rid>? ORDER BY rid',
                                 parameters + (last_rid,))
        n = get_state(db, 'documents', 0)
        while True:
            batch = cursor.fetchmany(10000)
            if not batch:
                break
            db.executemany('INSERT INTO fit(rowid,name,address) VALUES (?,?,?)', batch)
            n += len(batch)
            set_state(db, 'last_rid', batch[-1][0])
            set_state(db, 'documents', n)
            db.commit()
            if n % 100000 == 0:
                log(f'IDF fit: {n:,} training S1 documents')
        if not n:
            raise ValueError('No training entities available to fit IDF')
        db.execute('DELETE FROM idf')
        for route in ('name', 'address', 'combined'):
            cursor = (db.execute('SELECT term,doc FROM union_vocab') if route == 'combined'
                      else db.execute('SELECT term,doc FROM col_vocab WHERE col=?', (route,)))
            while True:
                batch = cursor.fetchmany(10000)
                if not batch:
                    break
                db.executemany('INSERT INTO idf VALUES (?,?,?)',
                               ((route, gram, math.log((n + 1) / (df + 1)) + 1) for gram, df in batch))
        set_state(db, 'documents', n)
        set_state(db, 'complete', True)
        db.commit()
    log(f'IDF ready: {n:,} training S1s; validation S1s excluded')
    return output_path


def load_idf(path):
    with closing(connect(path, True)) as db:
        if not get_state(db, 'complete', False):
            raise ValueError('IDF fit incomplete')
        result = {route: {} for route in ('name', 'address', 'combined')}
        for route, gram, value in db.execute('SELECT route,gram,value FROM idf'):
            result[route][gram] = value
        result['unseen'] = math.log(get_state(db, 'documents') + 1) + 1
        return result


def prepare_statistics(index_path):
    """Materialize exact pool DFs once; FTS vocabulary lookups decode posting lists."""
    with closing(connect(index_path)) as db:
        if not get_state(db,'complete',False):
            raise ValueError('Complete the pool index before preparing statistics')
        if get_state(db,'statistics_ready',False):
            return
        db.execute('CREATE TABLE IF NOT EXISTS statistics(kind TEXT,term TEXT,field TEXT,df INTEGER,'
                   'PRIMARY KEY(kind,term,field)) WITHOUT ROWID')
        for kind,table in (('gram','gram_vocab'),('word','word_vocab')):
            log(f'{Path(index_path).name}: storing {kind} document frequencies...')
            cursor=db.execute(f'SELECT term,col,doc FROM {table}')
            while True:
                batch=cursor.fetchmany(10000)
                if not batch:
                    break
                db.executemany('INSERT OR REPLACE INTO statistics VALUES (?,?,?,?)',
                               ((kind,term,field,df) for term,field,df in batch))
        set_state(db,'statistics_ready',True)
        db.commit()
