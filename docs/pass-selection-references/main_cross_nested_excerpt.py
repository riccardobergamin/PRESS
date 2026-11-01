class Reference:
    def f_obj_cross_nested(self, _, __):
        """
        Objective function of the binary classification boosters, for a cross-nested rumboost.

        Returns
        -------
        grad : numpy array
            The gradient with the cross-entropy loss function and cross-nested probabilities.
        hess : numpy array
            The hessian with the cross-entropy loss function and cross-nested probabilities (second derivative approximation rather than the hessian).
        """
        j = self._current_j  # jth booster
        if self.device is not None:
            if self.torch_compile:
                grad, hess = _f_obj_cross_nested_torch_compiled(
                    self.labels[self.subsample_idx],
                    self.preds_i_m,
                    self.preds_m,
                    self._preds,
                    self.num_classes,
                    self.mu,
                    self.device,
                    self.rum_structure[j]["utility"],
                )
            else:
                grad, hess = _f_obj_cross_nested_torch(
                    self.labels[self.subsample_idx],
                    self.preds_i_m,
                    self.preds_m,
                    self._preds,
                    self.num_classes,
                    self.mu,
                    self.device,
                    self.rum_structure[j]["utility"],
                )

            grad = grad.cpu().numpy()
            hess = hess.cpu().numpy()

            if self.subsample_idx.shape[0] < self.num_obs[0]:
                grad_rescaled = np.zeros(
                    (self.num_obs[0], len(self.rum_structure[j]["utility"]))
                )
                hess_rescaled = np.zeros(
                    (self.num_obs[0], len(self.rum_structure[j]["utility"]))
                )
                grad_rescaled[self.subsample_idx.cpu().numpy(), :] = grad.squeeze()
                hess_rescaled[self.subsample_idx.cpu().numpy(), :] = hess.squeeze()

                grad = grad_rescaled
                hess = hess_rescaled

            if not self.rum_structure[j]["shared"]:
                grad = grad.sum(axis=1)
                hess = hess.sum(axis=1)
            elif len(self.rum_structure[j]["variables"]) < len(
                self.rum_structure[j]["utility"]
            ):
                grad = grad.T.reshape(
                    int(
                        len(self.rum_structure[j]["utility"])
                        / len(self.rum_structure[j]["variables"])
                    ),
                    -1,
                ).sum(axis=0)
                hess = hess.T.reshape(
                    int(
                        len(self.rum_structure[j]["utility"])
                        / len(self.rum_structure[j]["variables"])
                    ),
                    -1,
                ).sum(axis=0)

            return grad.T.reshape(-1), hess.T.reshape(-1)

        j = self._current_j
        label = self.labels[self.subsample_idx]
        data_idx = np.arange(self.preds_i_m.shape[0])
        factor = self.num_classes / (self.num_classes - 1)

        pred_j_m = self.preds_i_m[
            :, self.rum_structure[j]["utility"], :
        ]  # pred of alternative j knowing nest m
        pred_i_m = self.preds_i_m[data_idx, label, :][
            :, None, :
        ]  # prediction of choice i knowing nest m
        pred_m = self.preds_m[:, None, :]  # prediction of choosing nest m
        pred_i = self._preds[data_idx, label][:, None, None]  # pred of choice i
        pred_j = self._preds[:, self.rum_structure[j]["utility"]][
            :, :, None
        ]  # pred of alt j

        pred_i_m_pred_m = pred_i_m * pred_m
        pred_j_m_pred_m = pred_j_m * pred_m
        pred_i_m_pred_i = pred_i_m * pred_i
        pred_i_m_squared = pred_i_m**2
        pred_j_m_squared = pred_j_m**2
        pred_i_squared = pred_i**2
        pred_j_m_pred_j_squared = (pred_j_m - pred_j) ** 2
        pred_i_m_1_mu_mu_pred_i = pred_i_m * (1 - self.mu) + self.mu - pred_i
        pred_j_m_1_mu_pred_j = pred_j_m * (1 - self.mu) - pred_j
        mu_squared = self.mu**2

        d_pred_i_Vi = np.sum(
            (pred_i_m_pred_m * pred_i_m_1_mu_mu_pred_i), axis=2, keepdims=True
        )  # first derivative of pred i with respect to Vi
        d_pred_i_Vj = np.sum(
            (pred_i_m_pred_m * pred_j_m_1_mu_pred_j), axis=2, keepdims=True
        )  # first derivative of pred i with respect to Vj
        d_pred_j_Vj = np.sum(
            (pred_j_m_pred_m * (pred_j_m_1_mu_pred_j + self.mu)),
            axis=2,
            keepdims=True,
        )  # first derivative of pred j with respect to Vj

        mu_3pim2_3pim_2pimpi_pi = self.mu * (
            -3 * pred_i_m_squared + 3 * pred_i_m + 2 * (pred_i_m_pred_i - pred_i)
        )
        pim2_2pimpi_pi2_dpiVi = (
            pred_i_m_squared - 2 * pred_i_m_pred_i + pred_i_squared - d_pred_i_Vi
        )
        mu2_2pim2_3pim_1 = mu_squared * (2 * pred_i_m_squared - 3 * pred_i_m + 1)
        mu2_pjm = mu_squared * (-pred_j_m)
        mu_pjm2_pjm = self.mu * (-pred_j_m_squared + pred_j_m)

        d2_pred_i_Vi = np.sum(
            (
                pred_i_m_pred_m
                * (mu2_2pim2_3pim_1 + mu_3pim2_3pim_2pimpi_pi + pim2_2pimpi_pi2_dpiVi)
            ),
            axis=2,
            keepdims=True,
        )
        d2_pred_i_Vj = np.sum(
            (
                pred_i_m_pred_m
                * (mu2_pjm + mu_pjm2_pjm + pred_j_m_pred_j_squared - d_pred_j_Vj)
            ),
            axis=2,
            keepdims=True,
        )

        mask = np.array(self.rum_structure[j]["utility"])[None, :] == label[:, None]
        grad = np.where(
            mask[:, :, None],
            ((-1 / pred_i) * d_pred_i_Vi),
            ((-1 / pred_i) * d_pred_i_Vj),
        )
        hess = np.where(
            mask[:, :, None],
            ((-1 / pred_i**2) * (d2_pred_i_Vi * pred_i - d_pred_i_Vi**2)),
            ((-1 / pred_i**2) * (d2_pred_i_Vj * pred_i - d_pred_i_Vj**2)),
        )
        hess *= factor

        if self.subsample_idx.size < self.num_obs[0]:
            grad_rescaled = np.zeros(
                (self.num_obs[0], len(self.rum_structure[j]["utility"]))
            )
            hess_rescaled = np.zeros(
                (self.num_obs[0], len(self.rum_structure[j]["utility"]))
            )
            grad_rescaled[self.subsample_idx, :] = grad.squeeze()
            hess_rescaled[self.subsample_idx, :] = hess.squeeze()

            grad = grad_rescaled
            hess = hess_rescaled

        if not self.rum_structure[j]["shared"]:
            grad = grad.sum(axis=1)
            hess = hess.sum(axis=1)
        elif len(self.rum_structure[j]["variables"]) < len(
            self.rum_structure[j]["utility"]
        ):
            grad = grad.T.reshape(
                int(
                    len(self.rum_structure[j]["utility"])
                    / len(self.rum_structure[j]["variables"])
                ),
                -1,
            ).sum(axis=0)
            hess = hess.T.reshape(
                int(
                    len(self.rum_structure[j]["utility"])
                    / len(self.rum_structure[j]["variables"])
                ),
                -1,
            ).sum(axis=0)

        grad = grad.T.reshape(-1)
        hess = hess.T.reshape(-1)

        return grad, hess
