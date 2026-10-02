# Business Entity Resolution — Code Queens

Pool-side assignment pipeline: every Source 2/3 record is linked to at most one
Source 1 entity (a structural property of the data, verified on the training ground
truth). Blocking retrieves the top-10 S1 candidates per pool record, a LightGBM
matcher scores them, and each pool record is assigned to its best candidate when
P(match) >= threshold (threshold tuned for exact macro F0.5 on held-out S1 entities).

## Environment
Python 3.10 (tested 3.10.11, Windows 11, 12 threads, 16 GB RAM), CPU only.

    pip install -r requirements.txt

## Reproduce end-to-end (data -> blocking -> matching -> output)

    bash run_pipeline.sh <path/to/dataset> <work_dir> <output_dir>

`<path/to/dataset>` is the provided `student_resource/dataset` (containing `train/`
and `test/`). Steps (all under `src/`):

| Step | Command | Output (in work_dir) | Time* |
|---|---|---|---|
| 1 normalize | `prep.py --split train/test` | `{split}_s{1,2,3}.pkl` | ~12 min/split |
| 2 blocking | `block.py --split train/test --k 10 --max-df 3000` | `{split}_cands.npz` | ~10 min/split |
| 3 train | `train.py --gt train_ground_truth.tsv --train-pool 900000 --trees 700 --out model` | `model/lgb.txt`, `model/decision.json` | ~50 min |
| 4 predict | `predict.py --model <work>/model --out <output_dir>` | `matching_results.tsv`, `candidate_pairs.tsv` | ~90 min |

*on the machine above. `train.py` prints blocking recall@k and the macro F0.5
threshold sweep on 3% held-out training S1 entities. `errors.py` prints an error
breakdown on those held-out predictions.

Put the work dir outside cloud-synced folders (intermediates are ~10 GB).
Validate with `python student_resource/utils/validate_submission.py`.

No external data, APIs or pretrained models are used.
