import numpy as np

from sklearn.decomposition import PCA

from jPCA.util import ensure_datas_is_list, preprocess
from jPCA.regression import skew_sym_regress

__all__ = ['JPCA']

class JPCA:
    def __init__(self, num_jpcs=2):
        assert num_jpcs % 2 == 0, "num_jpcs must be an even number."
        self.num_jpcs = num_jpcs
        self.jpcs = None

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
        def get_var_captured(projected_datas):
            X_full = np.concatenate(projected_datas)
            return np.diag(np.cov(X_full.T))
        assert self.jpcs is not None, "Model must be fit before calling jPCA.project."
        out = datas @ self.jpcs
        var_capt = get_var_captured(out)
        return out, var_capt

    @ensure_datas_is_list
    def fit(self,
            datas,
            pca=True,
            num_pcs=6, 
            subtract_cc_mean=True,
            tstart=0,
            tend=-1,
            times=None,
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
            pca: Boolean, whether or not we preprocess using PCA. Default True.
            num_pcs: Number of PCs to use when pca=True. Default 6.
            subtract_cc_mean: Whether we subtract CC mean during preprocessing.
            tstart: Starting time to use from the data. Default 0. 
            tend: Ending time to use from the data. -1 sets it to the end of the dataset.
            times: A list or numpy array containing the time for each time-bin of the data. 
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
            times = np.arange(T)
            tstart=0
            tend=-1

        processed_datas, full_data_var, pca_var_capt = \
            preprocess(datas, times, tstart=tstart, tend=tend, pca=pca,
                       subtract_cc_mean=subtract_cc_mean, num_pcs=num_pcs,
                       **preprocess_kwargs)

        # Estimate X dot via a first difference, and find the best
        # skew_symmetric matrix which explains the data.
        X = np.concatenate(processed_datas)
        X_dot = np.diff(X, axis=0)
        X = X[:-1]

        M_opt = skew_sym_regress(X, X_dot)
        self.jpcs = self._calculate_jpcs(M_opt)

        # Calculate the jpca basis using the eigenvectors of M_opt
        projected, jpca_var_capt = self.project(processed_datas)

        return (projected, 
                full_data_var,
                pca_var_capt,
                jpca_var_capt)
