import os
import time
import numpy as np

from tqdm import tqdm
from scipy.sparse import issparse
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor

from utils import knn, sinkhorn, cluster_metric, clustering_score


class AssignmentSequenceConstructor:
    """
    Construct the supervisory assignment sequence used by AutoSC.

    This module corresponds to the first stage of AutoSC:
        affinity graph -> sequential assignment construction.
    """

    def __init__(self, n_clusters, batch_size=16, n_trials=16, perplexity=200, sinkhorn_reg=0.2, sinkhorn_iter=4,
                 n_jobs=None):
        self.n_clusters = int(n_clusters)
        self.batch_size = int(batch_size)
        self.n_trials = int(n_trials)
        self.perplexity = int(perplexity)
        self.sinkhorn_reg = float(sinkhorn_reg)
        self.sinkhorn_iter = int(sinkhorn_iter)
        self.n_jobs = n_jobs

        # These variables are initialized when fit() is called.
        self.n_samples = None
        self.affinity_matrix = None
        self.reassign_interval = None

    def _assign(self, sample_idx, cluster_idx, assignments, cluster_sums, cluster_sizes, assigned_flags):
        # Assign selected samples to the given clusters.
        assignments[sample_idx] = cluster_idx

        # Update cluster_sums incrementally.
        # cluster_sums[i, k] stores the total affinity between sample i and the samples assigned to cluster k.
        for s_idx, c_idx in zip(sample_idx, cluster_idx):
            row = self.affinity_matrix[s_idx]
            cluster_sums[row.indices, c_idx] += row.data
            cluster_sizes[c_idx] += 1

        assigned_flags[sample_idx] = True

    def _select_next_batch(self, cluster_sums, cluster_sizes, assigned_flags, rng):
        # Compute soft assignments by Sinkhorn normalization.
        assignment_prob = sinkhorn(cluster_sums, reg=self.sinkhorn_reg, n_iter=self.sinkhorn_iter)

        # Each sample is assigned to the cluster with the largest soft assignment value.
        new_clusters = assignment_prob.argmax(axis=1)

        # Confidence score of each sample.
        # Lower confidence means the sample is harder to assign.
        sample_scores = np.nanmax(assignment_prob, axis=1)

        # Select only from unassigned samples.
        unassigned_indices = np.where(~assigned_flags)[0]
        batch_size = min(self.batch_size, unassigned_indices.size)

        if batch_size <= 0:
            return np.empty(0, dtype=int), np.empty(0, dtype=int)

        unassigned_scores = sample_scores[unassigned_indices]

        # Randomly shuffle before argpartition to break ties.
        perm = rng.permutation(len(unassigned_scores))
        shuffled_scores = unassigned_scores[perm]

        # Select the least confident samples.
        selected_rel_indices = np.argpartition(shuffled_scores, batch_size - 1)[:batch_size]
        selected_rel_indices = perm[selected_rel_indices]
        selected = unassigned_indices[selected_rel_indices]

        # If some clusters are still empty, fill them first to avoid invalid empty clusters.
        empty_clusters = np.flatnonzero(cluster_sizes == 0)
        k = min(batch_size, empty_clusters.size)

        selected_clusters = np.empty(batch_size, dtype=int)
        selected_clusters[:k] = empty_clusters[:k]
        selected_clusters[k:] = new_clusters[selected[k:]]

        return selected, selected_clusters

    def _reassign_samples(self, assignments, cluster_sums, cluster_sizes, assigned_flags):
        # Reassignment is skipped until all clusters have at least one sample.
        if cluster_sizes.min() == 0:
            return 0

        # Compute the best cluster for each sample according to average affinity.
        new_clusters = np.divide(cluster_sums, cluster_sizes, out=np.zeros_like(cluster_sums),
                                 where=cluster_sizes[None, :] > 0).argmax(axis=-1)

        # Only already assigned samples are considered for reassignment.
        idxs = np.flatnonzero(assigned_flags)
        new_clusters_assigned = new_clusters[idxs]
        old_clusters = assignments[idxs]

        move_mask = new_clusters_assigned != old_clusters
        samples_to_move = idxs[move_mask]

        if samples_to_move.size == 0:
            return 0

        from_clusters = assignments[samples_to_move]
        to_clusters = new_clusters_assigned[move_mask]

        # Move samples and update cluster statistics incrementally.
        for sample_idx, f, t in zip(samples_to_move, from_clusters, to_clusters):
            row = self.affinity_matrix[sample_idx]

            cluster_sums[row.indices, f] -= row.data
            cluster_sizes[f] -= 1

            cluster_sums[row.indices, t] += row.data
            cluster_sizes[t] += 1

        assignments[samples_to_move] = to_clusters

        return samples_to_move.size

    def trial(self, y=None, trial_id=0, seed=None):
        rng = np.random.default_rng(seed)

        n_samples = self.n_samples
        n_clusters = self.n_clusters
        batch_size = min(self.batch_size, n_samples)

        # cluster_sums records the affinity from each sample to each constructed cluster.
        cluster_sums = np.zeros((n_samples, n_clusters), dtype=np.float64)
        cluster_sizes = np.zeros(n_clusters, dtype=np.int32)

        # -1 means the sample has not been assigned yet.
        assignments = -np.ones(n_samples, dtype=np.int32)
        assigned_flags = np.zeros(n_samples, dtype=bool)
        selection_order = []

        # Randomly initialize the first batch.
        first_selected = rng.choice(n_samples, size=batch_size, replace=False)
        first_selected_clusters = rng.choice(n_clusters, size=batch_size, replace=batch_size > n_clusters)

        self._assign(first_selected, first_selected_clusters, assignments, cluster_sums, cluster_sizes, assigned_flags)
        selection_order.append(first_selected)

        # Number of remaining construction steps.
        n_steps = int(np.ceil((n_samples - batch_size) / batch_size))
        n_steps = max(n_steps, 0)

        sample_iter = range(n_steps)
        if trial_id == 0:
            sample_iter = tqdm(sample_iter, total=n_steps, ncols=130)

        n_changed = 0

        for iter_idx in sample_iter:
            # Select the next batch of hard samples and assign them.
            selected, selected_clusters = self._select_next_batch(cluster_sums, cluster_sizes, assigned_flags, rng)

            if selected.size == 0:
                break

            self._assign(selected, selected_clusters, assignments, cluster_sums, cluster_sizes, assigned_flags)

            # Periodically refine assigned samples by local reassignment.
            if iter_idx % self.reassign_interval == 0 or iter_idx == n_steps - 1:
                n_changed = self._reassign_samples(assignments, cluster_sums, cluster_sizes, assigned_flags)

            if trial_id == 0:
                sample_iter.set_postfix({"n_changed": n_changed, "cluster_sizes": cluster_sizes[:10]})

            selection_order.append(selected)

        # Safety check: assign any remaining samples if they were not selected.
        if not assigned_flags.all():
            remaining = np.flatnonzero(~assigned_flags)
            remaining_clusters = rng.integers(low=0, high=n_clusters, size=remaining.size)

            self._assign(remaining, remaining_clusters, assignments, cluster_sums, cluster_sizes, assigned_flags)
            selection_order.append(remaining)

        # Final assignment is obtained by assigning each sample to the cluster with the largest average affinity.
        final_assignments = np.divide(cluster_sums, cluster_sizes, out=np.zeros_like(cluster_sums),
                                      where=cluster_sizes[None, :] > 0).argmax(axis=-1)

        # Use the graph clustering score to select the best trial.
        score_ncut = clustering_score(assignments, self.affinity_matrix)

        selection_order = np.concatenate(selection_order)

        # Make sure each sample appears exactly once in the assignment sequence.
        _, unique_pos = np.unique(selection_order, return_index=True)
        unique_pos = np.sort(unique_pos)
        selection_order = selection_order[unique_pos]

        # Append missing samples if any numerical or indexing issue occurs.
        if selection_order.size != n_samples:
            missing = np.setdiff1d(np.arange(n_samples), selection_order)
            selection_order = np.concatenate([selection_order, missing])

        return final_assignments, score_ncut, selection_order

    def fit(self, X, y=None):
        self.n_samples = X.shape[0]

        # Reassignment frequency is adapted to the number of construction steps.
        self.reassign_interval = int(np.ceil(self.n_samples / self.batch_size / 1000))
        self.reassign_interval = max(self.reassign_interval, 1)

        # If X is already a sparse affinity matrix, use it directly.
        # Otherwise, construct the affinity matrix from input features.
        if issparse(X):
            self.affinity_matrix = X.tocsr()
        else:
            self.affinity_matrix = knn(X, self.perplexity)

        # Each trial uses a different random seed.
        seeds = np.random.randint(0, 1e8, size=self.n_trials, dtype=int)

        trial_assignments = np.empty((self.n_trials, self.n_samples), dtype=int)
        trial_scores = np.empty(self.n_trials, dtype=np.float64)
        trial_selection_order = np.empty((self.n_trials, self.n_samples), dtype=int)

        # Run a single trial directly.
        if self.n_trials == 1:
            assignment, score, selection_order = self.trial(y=y, trial_id=0, seed=seeds[0])

            trial_assignments[0] = assignment
            trial_scores[0] = score
            trial_selection_order[0] = selection_order

        # Run multiple trials in parallel and keep the best one.
        else:
            n_jobs = self.n_jobs
            if n_jobs is None:
                n_jobs = self.n_trials
            n_jobs = min(int(n_jobs), self.n_trials, os.cpu_count() or self.n_trials)

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = [executor.submit(self.trial, y, i, seeds[i]) for i in range(self.n_trials)]

                for i, fut in enumerate(tqdm(as_completed(futures), total=self.n_trials, desc="Trials", ncols=130)):
                    assignment, score, selection_order = fut.result()

                    trial_assignments[i] = assignment
                    trial_scores[i] = score
                    trial_selection_order[i] = selection_order

        # Select the trial with the best graph clustering score.
        best_idx = np.nanargmax(trial_scores)

        score = trial_scores[best_idx]
        assignments = trial_assignments[best_idx]
        selection_order = trial_selection_order[best_idx]
        return assignments, selection_order
