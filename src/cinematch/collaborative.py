"""Item-based collaborative filtering utilities."""

import numpy as np
import pandas as pd


def build_user_item_matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    """Create a user-by-item rating matrix from long-form ratings data.

    Expected input columns:
    - user_id
    - item_id
    - rating

    The returned matrix should use users as rows, movies as columns, and
    observed ratings as values. Missing user/movie pairs should stay missing
    instead of being treated as zero ratings.
    """
    matrix = ratings.pivot_table(
        index="user_id",
        columns="item_id",
        values="rating",
        aggfunc="mean",
    )

    return matrix.sort_index().sort_index(axis=1)


def center_user_ratings(user_item_matrix: pd.DataFrame) -> pd.DataFrame:
    """Subtract each user's mean rating from that user's observed ratings.

    User-centering removes the effect of generous or strict raters before item
    similarity is computed. Missing ratings should remain missing.
    """
    user_means = user_item_matrix.mean(axis=1)
    centered = user_item_matrix.sub(user_means, axis=0)

    return centered


def cosine_similarity(vector_a: pd.Series, vector_b: pd.Series) -> float:
    """Compute cosine similarity between two aligned rating vectors.

    The vectors should already be aligned on the same user index. Missing values
    should be handled consistently before the dot product is computed.
    """
    aligned = pd.concat([vector_a, vector_b], axis=1).dropna()
    values_a = aligned.iloc[:, 0].to_numpy()
    values_b = aligned.iloc[:, 1].to_numpy()

    denominator = np.linalg.norm(values_a) * np.linalg.norm(values_b)
    if denominator == 0:
        return 0.0

    return float(np.dot(values_a, values_b) / denominator)


def item_similarity_with_overlap(
    centered_matrix: pd.DataFrame,
    item_a: int,
    item_b: int,
) -> tuple[float, int]:
    """Compute item similarity and the number of common rating users."""
    vector_a = centered_matrix[item_a]
    vector_b = centered_matrix[item_b]

    aligned = pd.concat([vector_a, vector_b], axis=1).dropna()
    common_users = len(aligned)
    if common_users == 0:
        return 0.0, 0

    values_a = aligned.iloc[:, 0].to_numpy()
    values_b = aligned.iloc[:, 1].to_numpy()

    denominator = np.linalg.norm(values_a) * np.linalg.norm(values_b)
    if denominator == 0:
        return 0.0, common_users

    similarity = float(np.dot(values_a, values_b) / denominator)

    return similarity, common_users


def compute_item_similarity_and_overlap_matrices(
    centered_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute item-item cosine similarity and common-user overlap matrices."""
    item_ids = centered_matrix.columns
    values = centered_matrix.to_numpy(dtype=float)
    observed = ~np.isnan(values)
    observed_values = observed.astype(float)
    filled_values = np.nan_to_num(values, nan=0.0)

    overlap_values = observed_values.T @ observed_values
    dot_products = filled_values.T @ filled_values

    squared_values = filled_values**2
    left_norm_squares = squared_values.T @ observed_values
    denominators = np.sqrt(left_norm_squares * left_norm_squares.T)

    with np.errstate(divide="ignore", invalid="ignore"):
        similarity_values = dot_products / denominators
    similarity_values[denominators == 0] = 0.0

    similarity_matrix = pd.DataFrame(
        similarity_values,
        index=item_ids,
        columns=item_ids,
    )
    overlap_matrix = pd.DataFrame(
        overlap_values.astype(int),
        index=item_ids,
        columns=item_ids,
    )

    return similarity_matrix, overlap_matrix


def compute_item_similarity_matrix(centered_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute an item-by-item cosine similarity matrix.

    Rows and columns should both be movie item IDs. The diagonal should be 1.0
    for items with non-zero centered vectors.
    """
    similarity_matrix, _ = compute_item_similarity_and_overlap_matrices(
        centered_matrix
    )

    return similarity_matrix


def top_similar_items(
    similarity_matrix: pd.DataFrame,
    item_id: int,
    n: int = 10,
) -> pd.Series:
    """Return the top-n most similar items for one movie, excluding itself."""
    similarities = similarity_matrix.loc[item_id].drop(index=item_id)

    return similarities.sort_values(ascending=False).head(n)


def recommend_items_for_user(
    ratings: pd.DataFrame,
    centered_matrix: pd.DataFrame,
    user_id: int,
    n: int = 10,
    min_rating: float = 4.0,
    min_similarity: float = 0.0,
    min_common_users: int = 5,
) -> pd.Series:
    """Recommend unseen items similar to movies the user rated highly.

    The score is a similarity-weighted sum of the user's ratings for liked
    movies. This keeps the first recommendation step simple: recommend movies
    that look similar to what the user already liked, while excluding movies the
    user has already rated.
    """
    user_history = ratings.loc[ratings["user_id"] == user_id]
    if user_history.empty:
        raise ValueError(f"user_id {user_id} is not present in the ratings table.")

    liked_items = user_history.loc[user_history["rating"] >= min_rating]
    if liked_items.empty:
        return pd.Series(dtype=float, name="score")

    rated_items = set(user_history["item_id"])
    scores: dict[int, float] = {}

    for row in liked_items.itertuples(index=False):
        source_item_id = int(row.item_id)
        source_rating = float(row.rating)

        if source_item_id not in centered_matrix.columns:
            continue

        source_vector = centered_matrix[source_item_id]

        for candidate_item_id in centered_matrix.columns:
            candidate_item_id = int(candidate_item_id)
            if candidate_item_id in rated_items:
                continue

            candidate_vector = centered_matrix[candidate_item_id]
            aligned = pd.concat([source_vector, candidate_vector], axis=1).dropna()
            common_users = len(aligned)
            if common_users < min_common_users:
                continue

            similarity = cosine_similarity(source_vector, candidate_vector)
            if similarity <= min_similarity:
                continue

            scores[candidate_item_id] = (
                scores.get(candidate_item_id, 0.0) + similarity * source_rating
            )

    if not scores:
        return pd.Series(dtype=float, name="score")

    return pd.Series(scores, name="score").sort_values(ascending=False).head(n)


def predict_rating_for_user_item(
    ratings: pd.DataFrame,
    centered_matrix: pd.DataFrame,
    user_id: int,
    item_id: int,
    min_similarity: float = 0.0,
    min_common_users: int = 5,
    similarity_cache: dict[tuple[int, int], tuple[float, int]] | None = None,
) -> float:
    """Predict one user's rating for one item with item-based CF."""
    user_history = ratings.loc[ratings["user_id"] == user_id]
    if user_history.empty:
        raise ValueError(f"user_id {user_id} is not present in the ratings table.")

    if item_id not in centered_matrix.columns:
        raise ValueError(f"item_id {item_id} is not present in the ratings table.")

    user_mean = float(user_history["rating"].mean())
    numerator = 0.0
    denominator = 0.0

    for row in user_history.itertuples(index=False):
        source_item_id = int(row.item_id)
        source_rating = float(row.rating)

        if source_item_id == item_id:
            continue

        if source_item_id not in centered_matrix.columns:
            continue

        cache_key = tuple(sorted((item_id, source_item_id)))
        if similarity_cache is not None and cache_key in similarity_cache:
            similarity, common_users = similarity_cache[cache_key]
        else:
            similarity, common_users = item_similarity_with_overlap(
                centered_matrix=centered_matrix,
                item_a=item_id,
                item_b=source_item_id,
            )
            if similarity_cache is not None:
                similarity_cache[cache_key] = (similarity, common_users)

        if common_users < min_common_users:
            continue

        if similarity <= min_similarity:
            continue

        centered_rating = source_rating - user_mean
        numerator += similarity * centered_rating
        denominator += abs(similarity)

    if denominator == 0:
        return user_mean

    prediction = user_mean + numerator / denominator

    return float(np.clip(prediction, 1.0, 5.0))


def predict_ratings_item_based(
    train_ratings: pd.DataFrame,
    rows_to_predict: pd.DataFrame,
    min_similarity: float = 0.0,
    min_common_users: int = 5,
) -> pd.Series:
    """Predict ratings for multiple user-item rows with item-based CF."""
    user_item_matrix = build_user_item_matrix(train_ratings)
    centered_matrix = center_user_ratings(user_item_matrix)
    global_mean = float(train_ratings["rating"].mean())
    user_means = train_ratings.groupby("user_id")["rating"].mean()
    similarity_matrix, overlap_matrix = compute_item_similarity_and_overlap_matrices(
        centered_matrix
    )
    user_histories = {
        int(user_id): user_history
        for user_id, user_history in train_ratings.groupby("user_id")
    }

    predictions = []

    for row in rows_to_predict.itertuples(index=False):
        user_id = int(row.user_id)
        item_id = int(row.item_id)

        if user_id not in user_histories or item_id not in similarity_matrix.index:
            prediction = float(user_means.get(user_id, global_mean))
        else:
            prediction = _predict_rating_from_precomputed_matrices(
                user_history=user_histories[user_id],
                similarity_matrix=similarity_matrix,
                overlap_matrix=overlap_matrix,
                user_mean=float(user_means.loc[user_id]),
                item_id=item_id,
                min_similarity=min_similarity,
                min_common_users=min_common_users,
            )

        predictions.append(prediction)

    return pd.Series(
        predictions,
        index=rows_to_predict.index,
        name="prediction",
    )


def _predict_rating_from_precomputed_matrices(
    user_history: pd.DataFrame,
    similarity_matrix: pd.DataFrame,
    overlap_matrix: pd.DataFrame,
    user_mean: float,
    item_id: int,
    min_similarity: float,
    min_common_users: int,
) -> float:
    """Predict one rating using precomputed item similarity and overlap."""
    usable_history = user_history.loc[
        (user_history["item_id"] != item_id)
        & (user_history["item_id"].isin(similarity_matrix.columns))
    ]
    if usable_history.empty:
        return user_mean

    usable_history = usable_history.set_index("item_id")
    source_item_ids = usable_history.index

    similarities = similarity_matrix.loc[item_id, source_item_ids].to_numpy(
        dtype=float
    )
    overlaps = overlap_matrix.loc[item_id, source_item_ids].to_numpy(dtype=int)
    centered_ratings = usable_history["rating"].to_numpy(dtype=float) - user_mean

    usable_mask = (overlaps >= min_common_users) & (similarities > min_similarity)
    if not usable_mask.any():
        return user_mean

    usable_similarities = similarities[usable_mask]
    usable_centered_ratings = centered_ratings[usable_mask]

    denominator = np.abs(usable_similarities).sum()
    if denominator == 0:
        return user_mean

    prediction = user_mean + (
        usable_similarities * usable_centered_ratings
    ).sum() / denominator

    return float(np.clip(prediction, 1.0, 5.0))
