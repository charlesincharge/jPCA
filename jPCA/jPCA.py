from typing import Optional

import numpy as np

from sklearn.decomposition import PCA

from jPCA.util import ensure_datas_is_list
from jPCA.regression import skew_sym_regress

__all__ = ['JPCA']

class JPCA:
    def __init__(
            self,
            num_pcs: Optional[int] = 6,
            num_jpcs: int = 2,
            soft_normalize: float = 5,
            subtract_cc_mean: bool = True,
    ):
        """

        Parameters
        ----------
        num_jpcs: number of jPCA components to calculate. Must be even for plotting.
        num_pcs: number of principal components to use when preprocessing with PCA.
            Set to `None` to not use PCA.
        soft_normalize: Constant used during soft-normalization preprocessing step.
                        Adapted from original jPCA matlab code. Normalized firing rate is
                        computed by dividing by the range of the unit across all conditions and times
                        plus the soft_normalize constant: Y_{cij} = (max(Y_{:i:}) - min(Y_{:i:}) + C)
                        where Y_{cij} is the cth condition, ith neuron, at the jth time bin.
                        C is the constant provided by soft_normalize. Set C negative to skip the
                        soft-normalizing step.
        subtract_cc_mean: Whether to subtract the mean across conditions.
        """
        assert num_jpcs % 2 == 0, "num_jpcs must be an even number."
        assert soft_normalize >= 0, "soft_normalize must be non-negative."
        # Initialize attributes with arguments
        self.num_jpcs = num_jpcs
        self.pca = PCA(n_components=num_pcs) if (num_pcs is not None) else None
        self.soft_normalize = soft_normalize
        self.subtract_cc_mean = subtract_cc_mean

        # Initialize attributes that will be set later
        self.jpcs = None
        self.is_fitted : bool = False
        self.cc_mean_ : Optional[float] = None
        self.firing_rate_range_ : Optional[float] = None

    def _calculate_jpcs(self, M):
        num_jpcs = self.num_jpcs
        D, _ = M.shape

        # Eigenvalues are not necessarily sorted
        eigvals, eigvecs = np.linalg.eig(M)
        idx = np.argsort(np.abs(np.imag(eigvals)))[::-1]
        eigvecs = eigvecs[:, idx]
        eigvals = eigvals[idx]

        jpca_basis = np.zeros((D, num_jpcs))
        for k in range(0, num_jpcs, 2):
            # One eigenvalue will have negative imaginary component,
            # the other will have positive imaginary component.
            if np.imag(eigvals[k]) > 0:
                v1 = eigvecs[:, k] + eigvecs[:, k+1]
                v2 = -np.imag(eigvecs[:, k] - eigvecs[:, k+1])
            else:
                v1 = eigvecs[:, k] + eigvecs[:, k+1]
                v2 = -np.imag(eigvecs[:, k+1] - eigvecs[:, k])

            V = np.real(np.column_stack((v1, v2))) / np.sqrt(2)
            jpca_basis[:, k:k+2] = V
        return jpca_basis

    def align_jpcs(self, datas):
        """
        Given an input list of datas, rotate each pair of jPCs
        such that the preparatory states tend to lie along the X-axis.
        This mirrors the functionality of the MATLAB codepack.
        """
        projected, _ = self.project(datas, )
        for k in range(0, self.num_jpcs, 2): 

            # The X and Y axis (for plotting) are the standard basis. We are
            # effectively rotating this basis to align with the prep states.
            # Note that this is purely for aesthetics -- it does not change the
            # amount of variance captured by a given jPCA plane.
            basis = np.array([[1, 0], [0, 1]])
            prep_states = np.row_stack([x[0, k:k+2] for x in projected])
            pca = PCA(n_components=2)
            pca = pca.fit(prep_states)
            rot = pca.components_

            pc1 = np.append(rot[0, :], 0)
            pc2 = np.append(rot[1, :], 0)
            cross = np.cross(pc1, pc2)
            if cross[2] < 0:
                rot[1, :] = -rot[1, :]
            basis = basis @ rot.T
            test_proj = np.concatenate([x[:, k:k+2] @ basis for x in projected])
            if np.max(np.abs(test_proj[:, 1])) > np.max(test_proj[:, 1]):
                basis = -basis
            
            # Update jpc pair
            self.jpcs[:, k:k+2] = self.jpcs[:, k:k+2] @ basis

    def preprocess(
            self,
            datas,
            times,
            tstart=-50,
            tend=150,
            ):
        """
        Preprocess data for jPCA.

        Args
        ----
            datas: List of trials, where each element of the list has shape Times x Neurons.
                As an example, this might be the output of load_churchland_data()

            times: List of times for the experiment. Typically, time zero corresponds
                to a stimulus onset. This list is used for extracting the set of
                data to be analyzed (see tstart and tend args).

            tstart: Integer. Starting time for analysis. For example, if times is [-10, 0 , 10]
                    and tstart=0, then the data returned by this function will start
                    at index 1.

            tend: Integer. Ending time for analysis.

        Returns
        -------
            data_list: List of arrays, each T x N. T will depend on the values
                passed for tstart and tend. N will be equal to num_pcs if pca=True.
                Otherwise the dimension of the data will remain unchanged.
            full_data_variance: Float, variance of original dataset.
            pca_variance_captured: Array, variance captured by each PC.
                                   If pca=False, this is set to None.
        """
        datas = np.stack(datas, axis=0)
        num_conditions, num_time_bins, num_units = datas.shape

        if self.soft_normalize > 0:
            self.firing_rate_range_ = np.max(datas, axis=(0,1)) - np.min(datas, axis=(0,1))
            datas /= (self.firing_rate_range_ + self.soft_normalize)

        if self.subtract_cc_mean:
            self.cc_mean_ = np.mean(datas, axis=0, keepdims=True)
            datas -= self.cc_mean_

        # For consistency with the original jPCA matlab code,
        # we compute PCA using only the analyzed times.
        idx_start = times.index(tstart)
        idx_end = times.index(tend) + 1 # Add one so idx is inclusive
        num_time_bins = idx_end - idx_start
        datas = datas[:, idx_start:idx_end, :]

        # Reshape to perform PCA on all trials at once.
        X_full = np.concatenate(datas)
        full_data_var = np.sum(np.var(X_full, axis=0))
        pca_variance_captured = None

        if self.pca:
            datas = self.pca.fit_transform(X_full)
            datas = datas.reshape(num_conditions, num_time_bins, self.pca.n_components_
            pca_variance_captured = self.pca.explained_variance_

        data_list = [x for x in datas]
        return data_list, full_data_var, pca_variance_captured

    @ensure_datas_is_list
    def project(self, datas):
        """
        Project data into dimensions which capture rotations.
        We assume that rotation_matrix is an orthogonal matrix which was found
        by fitting a rotational LDS to data (e.g via setting dynamics="rotational").
        This function then projects on the top eigenvectors of this rotation matrix.


        Returns
        -------
        out: T x num_components project of the data, which should capture rotations
        """
        assert self.jpcs is not None, "Model must be fit before calling jPCA.project."
        proj = datas @ self.jpcs
        var_capt = np.var(proj, axis=(0,1))
        return [x for x in proj], var_capt

    # def: np.var(x, axis):
    #   mean = np.mean(x, axis=axis, keepdims=True)
    #   x_bar = x - mean
    #   sq_err = np.sum(x_bar**2, axis=axis)
    #   return sq_err / x.shape[axis]

    @ensure_datas_is_list
    def fit(self,
            datas,
            tstart=0,
            tend=-1,
            times=None,
            align_axes_to_data=True,
            **preprocess_kwargs):
    
        """
        Fit jPCA to a dataset. This function does not do plotting -- 
        that is handled separately.
        See "args" section for information on data formatting.
        The tstart and tend arguments are used to exclude certain parts of the data
        during fitting.

        For example, if tstart=-20, tend=10, and times = [-30, -20, -10, 0, 10, 20],
        each trial would include the 2nd through 5th timebins (the other timebins)
        would be discarded. By default the entire dataset is used.

        Args
        ----
            datas: A list containing trials. Each element of the list should be T x D,
                where T is the length of the trial, and D is the number of neurons.
            tstart: Starting time to use from the data. Default 0.
            tend: Ending time to use from the data. -1 sets it to the end of the dataset.
            times: A list containing the time for each time-bin of the data.
                This is used to determine which time bins are included and excluded
                from the data.]

        Returns
        -------
            projected: A list of trials projected onto the jPCS. Each entry is an array
                    with shape T x num_jpcs.
            full_data_variance: Float, variance of the full dataset, for calculating
                                variance captured by projections.
            pca_captured_variance: Array, size is num_pcs. Contains variance captured
                                by each PC.
            jpca_captured_variance: Array, size is num_jpcs. Contains variance captured 
                                    by each jPC.
        """
        assert isinstance(datas, list), "datas must be a list."
        assert datas, "datas cannot be empty"
        T = datas[0].shape[1]
        if times is None:
            times = list(range(T))
            tstart=0
            tend=-1

        processed_datas, full_data_var, pca_var_capt = \
            self.preprocess(
                datas,
                times,
                tstart=tstart,
                tend=tend,
                **preprocess_kwargs
            )
        self.full_data_var = full_data_var

        # Estimate X dot via a first difference, and find the best
        # skew_symmetric matrix which explains the data.
        X = np.concatenate([x[:-1] for x in processed_datas])
        X_dot = np.concatenate([np.diff(x, axis=0) for x in processed_datas])

        # Calculate the jpca basis using the eigenvectors of M_opt
        M_opt = skew_sym_regress(X, X_dot)
        self.M_skew = M_opt
        self.jpcs = self._calculate_jpcs(M_opt)

        # Optionally align axes so data is spread along horizontal axes in plots
        if align_axes_to_data:
            self.align_jpcs(processed_datas)

        projected, jpca_var_capt = self.project(processed_datas)

        self.is_fitted = True
        return (projected,
                full_data_var,
                pca_var_capt,
                jpca_var_capt)
