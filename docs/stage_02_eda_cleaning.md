# Stage 02 - EDA and Basic Cleaning

This development note guides the next data stage. It is intentionally
developer-facing and can be compressed into `README.md` or removed before the
final portfolio submission.

## Goal

Build a reusable data profiling layer before modeling. The output should help
answer:

- Are the loaded MovieLens tables structurally complete?
- What are the basic rating, user, movie, and timestamp distributions?
- Are there duplicates, missing values, or suspicious sparsity patterns?
- What facts should guide train/test splitting and first baseline models?

## Scope

Create reusable summary functions in `src/cinematch/eda.py`, then expose them
through a repeatable script such as `scripts/profile_data.py`.

Keep notebook exploration local. Move only reusable logic into `src`.

## Student-Owned Tasks

1. Create `src/cinematch/eda.py`.
2. Implement `summarize_ratings(ratings: pd.DataFrame) -> dict[str, object]`.
3. Implement `summarize_users(users: pd.DataFrame) -> dict[str, object]`.
4. Implement `summarize_movies(movies: pd.DataFrame) -> dict[str, object]`.
5. Implement `build_data_profile(...) -> dict[str, object]` that combines all
   table summaries.
6. Create `scripts/profile_data.py` to load, validate, profile, and print the
   key summary values.
7. Add unit tests with small fake DataFrames before using the full dataset.

## Suggested Metrics

### Ratings

- row count
- unique user count
- unique item count
- rating min, max, mean, median
- rating distribution by score
- duplicate `(user_id, item_id, timestamp)` count
- timestamp min and max, converted to readable datetimes
- ratings per user: min, max, mean, median
- ratings per movie: min, max, mean, median

### Users

- row count
- unique user count
- age min, max, mean, median
- gender distribution
- occupation distribution
- duplicate user ID count

### Movies

- row count
- unique movie count
- duplicate movie ID count
- missing title count
- genre column totals
- movies with zero genre flags
- movies with multiple genre flags

## Implementation Hints

TODO(student): Write the function bodies yourself. Keep each function small and
return plain Python dictionaries so the output is easy to test and print.

HINT: Start with `.shape`, `.nunique()`, `.min()`, `.max()`, `.mean()`,
`.median()`, `.value_counts()`, `.duplicated()`, and `.groupby(...).size()`.

HINT: Use `pd.to_datetime(ratings["timestamp"], unit="s")` when converting
MovieLens Unix timestamps.

HINT: For genre columns, reuse the movie genre column names already defined in
`src/cinematch/data.py` instead of typing them again.

HINT: For tests, use tiny fake DataFrames. Do not make every test depend on the
full MovieLens dataset.

## References

REFERENCE: MovieLens schema and dataset facts:
`data/raw/ml-100k/README`

REFERENCE: Current project parsing and validation boundary:
`src/cinematch/data.py`

REFERENCE: Machine learning iteration mindset:
`/Users/qianjiangyue/Desktop/Study Zone/知识点汇总/Advanced Learning Algorithms/Develop a Machine Learning System（开发机器学习系统）/Iterative Loop of Machine Learning Development（机器学习的迭代发展）.md`

REFERENCE: Outlier and unusual-behavior thinking for later user analysis:
`/Users/qianjiangyue/Desktop/Study Zone/知识点汇总/Unsupervised learning recommenders reinforcement learning/Anomaly Detection（异常检测）/Anomaly Detection（异常检测）.md`

REFERENCE: Feature scale awareness for later modeling:
`/Users/qianjiangyue/Desktop/Study Zone/知识点汇总/Supervised Machine Learning Regression and Classification/Feature Engineering（特征处理）/Feature Scaling（特征缩放）.md`

REFERENCE: Course plotting patterns for later report figures:
`/Users/qianjiangyue/Desktop/Study Zone/示例及练习/第二课 Advanced Learning Algorithms/week3/8.Practice Lab Advice for applying machine learning/C2_W3_Assignment.ipynb`

## Acceptance Checks

Run these before committing this stage:

```bash
python scripts/validate_data.py
python scripts/profile_data.py
python -m pytest -q
python -m ruff check .
```

The profile script should print a compact summary, not raw tables.
