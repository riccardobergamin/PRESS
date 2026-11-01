class Reference:
    def f_obj_cross_nested(
            self,
            _,
            train_set: Dataset
        ):
            """
            Objective function of the binary classification boosters, for a cross-nested rumboost.

            Parameters
            ----------
            train_set : Dataset
                Training set used to train the jth booster. It means that it is not the full training set but rather another dataset containing the relevant features for that utility. It is the jth dataset in the RUMBoost object.
            Returns
            -------
            grad : numpy array
                The gradient with the cross-entropy loss function and cross-nested probabilities.
            hess : numpy array
                The hessian with the cross-entropy loss function and cross-nested probabilities (second derivative approximation rather than the hessian).
            """
            j = self._current_j
            labels = self.labels.astype(int)
            mu = np.array(self.mu).reshape(1, len(self.mu))
            data_idx = np.arange(self.preds_i_m.shape[0])
            factor = self.num_classes/(self.num_classes-1) #factor to correct redundancy (see Friedmann, Greedy Function Approximation)

            pred_j_m = self.preds_i_m[:,j,:] #pred of alternative j knowing nest m
            pred_i_m = self.preds_i_m[data_idx,labels,:] #prediction of choice i knowing nest m
            pred_m = self.preds_m[:,j,:] #prediction of choosing nest m
            pred_i = self._preds[data_idx,labels].reshape(-1, 1) #pred of choice i
            pred_j = self._preds[:,j].reshape(-1, 1) #pred of alt j

            d_pred_i_Vi = np.sum((pred_i_m * pred_m * (pred_i_m * (1 - mu) + mu - pred_i)), axis=1, keepdims=True) #first derivative of pred i with resepct to Vi
            d_pred_i_Vj = np.sum((pred_i_m * pred_m * (pred_j_m * (1 - mu) - pred_j)), axis=1, keepdims=True) #first derivative of pred i with resepct to Vj
            d_pred_j_Vj = np.sum((pred_j_m * pred_m * (pred_j_m * (1 - mu) + mu - pred_j)), axis=1, keepdims=True) #first derivative of pred j with resepct to Vj
            d2_pred_i_Vi = np.sum((pred_i_m * pred_m * (mu**2 * (2*pred_i_m**2 - 3*pred_i_m + 1) + mu * (-3*pred_i_m**2 + 3*pred_i_m + 2*pred_i*(pred_i_m-1)) + (pred_i_m**2 - 2*pred_i_m*pred_i + pred_i**2 - d_pred_i_Vi))), axis=1, keepdims=True)
            d2_pred_i_Vj = np.sum((pred_i_m * pred_m * (mu**2 * (-pred_j_m) + mu * (-pred_j_m**2 + pred_j_m) + (pred_j_m - pred_j)**2 - d_pred_j_Vj)), axis=1, keepdims=True)

            #two cases: 1. alt j is choice i, 2. alt j is not choice i
            grad = (labels == j).reshape(-1, 1) * (-1/pred_i) * d_pred_i_Vi + \
                   (1 - (labels == j)).reshape(-1, 1) * (-1/pred_i) * d_pred_i_Vj
            hess = (labels == j).reshape(-1, 1) * (-1/pred_i**2) * (d2_pred_i_Vi*pred_i - d_pred_i_Vi**2) + \
                   (1 - (labels == j)).reshape(-1, 1) * (-1/pred_i**2) * (d2_pred_i_Vj*pred_i - d_pred_i_Vj**2)

            hess = factor * hess

            return grad.reshape(-1), hess.reshape(-1)
