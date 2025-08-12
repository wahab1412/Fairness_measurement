from densratio import densratio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Code used based on https://github.com/wahab1412/Fairness-Measurement-ML and https://github.com/wahab1412/housing_fairness
# Metrics are independence and separation. Code source https://dalex.drwhy.ai/python-dalex-fairness-regression.html based on: Fairness Measures for Regression via Probabilistic Classification https://arxiv.org/pdf/2001.06089.pdf


def calc_fairness_LR_dict(y, y_hat, protected, privileged):    

    unique_protected = np.unique(protected)
    unique_unprivileged = unique_protected[unique_protected != privileged]

    data = pd.DataFrame(columns=['subgroup', 'independence', 'separation', 'sufficiency'])
    for unprivileged in unique_unprivileged:
        # filter elements
        array_elements = np.isin(protected, [privileged, unprivileged])

        y_u = ((y[array_elements] - y[array_elements].mean()) / y[array_elements].std()).reshape(-1, 1)
        s_u = ((y_hat[array_elements] - y_hat[array_elements].mean()) / y_hat[array_elements].std()).reshape(-1, 1)


        a = np.where(protected[array_elements] == privileged, 1, 0)

        p_s = LogisticRegression(penalty='none',random_state=0)
        p_ys = LogisticRegression(penalty='none',random_state=0)
        p_y = LogisticRegression(penalty='none',random_state=0)
        
        p_s.fit(s_u, a)
        p_y.fit(y_u, a)
        p_ys.fit(np.c_[y_u, s_u], a)

        pred_p_s = p_s.predict_proba(s_u.reshape(-1, 1))[:, 1]
        pred_p_y = p_y.predict_proba(y_u.reshape(-1, 1))[:, 1]
        pred_p_ys = p_ys.predict_proba(np.c_[y_u, s_u])[:, 1]
        
        n = len(a)
    
        r_ind = ((n - a.sum()) / a.sum()) * (pred_p_s / (1 - pred_p_s)).mean()
        r_sep = ((pred_p_ys / (1 - pred_p_ys) * (1 - pred_p_y) / pred_p_y)).mean()
        r_suf = ((pred_p_ys / (1 - pred_p_ys)) * ((1 - pred_p_s) / pred_p_s)).mean()

        independence = {
            'indep' : r_ind,
            'prob' : pred_p_s,
            'n' : n,
            'a_sum' : sum(a)
        }

        separation = {
            'sep' : r_sep,
            'pred_p_ys' : pred_p_ys,
            'pred_p_y' : pred_p_y
        }

        sufficiency = {
            'suf': r_suf,
            'pred_p_ys' : pred_p_ys,
            'pred_p_s' : pred_p_s
        }
        return independence,separation,sufficiency

        to_append = pd.DataFrame({'subgroup': [unprivileged],
                                'independence': [r_ind],
                                'separation': [r_sep],
                                'sufficiency': [r_suf]})

        data = pd.concat([data, to_append])

    to_append = pd.DataFrame({'subgroup': [privileged],
                            'independence': [1],
                            'separation': [1],
                            'sufficiency': [1]})

    data.index = data.subgroup
    data = data.iloc[:, 1:]
    return data

def calc_fairness_Ridge_dict(y, y_hat, protected, privileged):    

    unique_protected = np.unique(protected)
    unique_unprivileged = unique_protected[unique_protected != privileged]

    data = pd.DataFrame(columns=['subgroup', 'independence', 'separation', 'sufficiency'])
    for unprivileged in unique_unprivileged:
        # filter elements
        array_elements = np.isin(protected, [privileged, unprivileged])

        y_u = ((y[array_elements] - y[array_elements].mean()) / y[array_elements].std()).reshape(-1, 1)
        s_u = ((y_hat[array_elements] - y_hat[array_elements].mean()) / y_hat[array_elements].std()).reshape(-1, 1)


        a = np.where(protected[array_elements] == privileged, 1, 0)

        p_s = LogisticRegression(penalty='l2',random_state=0)
        p_ys = LogisticRegression(penalty='l2',random_state=0)
        p_y = LogisticRegression(penalty='l2',random_state=0)
        
        p_s.fit(s_u, a)
        p_y.fit(y_u, a)
        p_ys.fit(np.c_[y_u, s_u], a)

        pred_p_s = p_s.predict_proba(s_u.reshape(-1, 1))[:, 1]
        pred_p_y = p_y.predict_proba(y_u.reshape(-1, 1))[:, 1]
        pred_p_ys = p_ys.predict_proba(np.c_[y_u, s_u])[:, 1]
        
        n = len(a)
    
        r_ind = ((n - a.sum()) / a.sum()) * (pred_p_s / (1 - pred_p_s)).mean()
        r_sep = ((pred_p_ys / (1 - pred_p_ys) * (1 - pred_p_y) / pred_p_y)).mean()
        r_suf = ((pred_p_ys / (1 - pred_p_ys)) * ((1 - pred_p_s) / pred_p_s)).mean()

        independence = {
            'indep' : r_ind,
            'prob' : pred_p_s,
            'n' : n,
            'a_sum' : sum(a)
        }

        separation = {
            'sep' : r_sep,
            'pred_p_ys' : pred_p_ys,
            'pred_p_y' : pred_p_y
        }

        sufficiency = {
            'suf': r_suf,
            'pred_p_ys' : pred_p_ys,
            'pred_p_s' : pred_p_s
        }
        return independence,separation,sufficiency

        to_append = pd.DataFrame({'subgroup': [unprivileged],
                                'independence': [r_ind],
                                'separation': [r_sep],
                                'sufficiency': [r_suf]})

        data = pd.concat([data, to_append])

    to_append = pd.DataFrame({'subgroup': [privileged],
                            'independence': [1],
                            'separation': [1],
                            'sufficiency': [1]})

    data.index = data.subgroup
    data = data.iloc[:, 1:]
    return data


def calc_fairness_Lasso_dict(y, y_hat, protected, privileged):    

    unique_protected = np.unique(protected)
    unique_unprivileged = unique_protected[unique_protected != privileged]

    data = pd.DataFrame(columns=['subgroup', 'independence', 'separation', 'sufficiency'])
    for unprivileged in unique_unprivileged:
        # filter elements
        array_elements = np.isin(protected, [privileged, unprivileged])

        y_u = ((y[array_elements] - y[array_elements].mean()) / y[array_elements].std()).reshape(-1, 1)
        s_u = ((y_hat[array_elements] - y_hat[array_elements].mean()) / y_hat[array_elements].std()).reshape(-1, 1)


        a = np.where(protected[array_elements] == privileged, 1, 0)
        p_s = LogisticRegression(penalty='l1',solver='saga',random_state=0)
        p_ys = LogisticRegression(penalty='l1',solver='saga',random_state=0)
        p_y = LogisticRegression(penalty='l1',solver='saga',random_state=0)

        p_s.fit(s_u, a)
        p_y.fit(y_u, a)
        p_ys.fit(np.c_[y_u, s_u], a)

        pred_p_s = p_s.predict_proba(s_u.reshape(-1, 1))[:, 1]
        pred_p_y = p_y.predict_proba(y_u.reshape(-1, 1))[:, 1]
        pred_p_ys = p_ys.predict_proba(np.c_[y_u, s_u])[:, 1]
        
        n = len(a)
    
        r_ind = ((n - a.sum()) / a.sum()) * (pred_p_s / (1 - pred_p_s)).mean()
        r_sep = ((pred_p_ys / (1 - pred_p_ys) * (1 - pred_p_y) / pred_p_y)).mean()
        r_suf = ((pred_p_ys / (1 - pred_p_ys)) * ((1 - pred_p_s) / pred_p_s)).mean()

        independence = {
            'indep' : r_ind,
            'prob' : pred_p_s,
            'n' : n,
            'a_sum' : sum(a)
        }

        separation = {
            'sep' : r_sep,
            'pred_p_ys' : pred_p_ys,
            'pred_p_y' : pred_p_y
        }

        sufficiency = {
            'suf': r_suf,
            'pred_p_ys' : pred_p_ys,
            'pred_p_s' : pred_p_s
        }
        return independence,separation,sufficiency

        to_append = pd.DataFrame({'subgroup': [unprivileged],
                                'independence': [r_ind],
                                'separation': [r_sep],
                                'sufficiency': [r_suf]})

        data = pd.concat([data, to_append])

    to_append = pd.DataFrame({'subgroup': [privileged],
                            'independence': [1],
                            'separation': [1],
                            'sufficiency': [1]})

    data.index = data.subgroup
    data = data.iloc[:, 1:]
    return data



# Code https://github.com/hoxo-m/densratio_py

def calculate_regression_measures_density(y, y_hat, protected, privileged):    

    unique_protected = np.unique(protected)
    unique_unprivileged = unique_protected[unique_protected != privileged]

    l = []
    data_list = []
    alphas = [0, 0.25, 0.5, 0.75,1]
    for alpha in alphas:
        data = pd.DataFrame(columns=['subgroup', 'independence', 'separation', 'sufficiency'])

        for unprivileged in unique_unprivileged:
            # filter elements
            array_elements = np.isin(protected, [privileged, unprivileged])

            y_u = ((y[array_elements] - y[array_elements].mean()) / y[array_elements].std()).reshape(-1, 1)
            s_u = ((y_hat[array_elements] - y_hat[array_elements].mean()) / y_hat[array_elements].std()).reshape(-1, 1)

            a = np.where(protected[array_elements] == privileged, 1, 0)
            x_all_s_u = s_u
            x_1_s_u = s_u[a == 1]
            x_0_s_u = s_u[a == 0]



            x_all_y_u = y_u
            x_1_y_u = y_u[a == 1]
            x_0_y_u = y_u[a == 0]
            
            # separation
            densratio_obj = densratio(x_1_s_u, x_0_s_u, alpha=alpha, verbose=False )
            ind = densratio_obj.compute_density_ratio(x_all_s_u)
            r_ind = np.mean(ind)

            # separation
            densratio_obj1 = densratio(np.c_[x_1_y_u,x_1_s_u], np.c_[x_0_y_u,x_0_s_u], alpha=alpha, verbose=False)
            s1 = densratio_obj1.compute_density_ratio(np.c_[x_all_y_u,x_all_s_u])

            densratio_obj2 = densratio(x_1_y_u,x_0_y_u, alpha=alpha, verbose=False ) 
            s2 = densratio_obj2.compute_density_ratio(x_all_y_u)

            sep = s1 * s2 
            r_sep = np.mean(sep)

            # sufficiency
            densratio_obj1 = densratio(np.c_[x_1_y_u,x_1_s_u], np.c_[x_0_y_u,x_0_s_u], alpha=alpha, verbose=False )
            s1 = densratio_obj1.compute_density_ratio(np.c_[x_all_y_u,x_all_s_u])

            densratio_obj2 = densratio(x_1_s_u,x_0_s_u, alpha=alpha, verbose=False ) 
            s2 = densratio_obj2.compute_density_ratio(x_all_s_u)

            suf = s1 * s2 
            r_suf = np.mean(suf)
            
            res = { 'alpha':alpha,
                   'r_ind' : r_ind,
                   'ind' : ind,
                   'r_sep' : r_sep,
                   'sep' : sep,
                   'r_suf' : r_suf,
                   'suf' : suf
            }
            l.append(res)
        
            to_append = pd.DataFrame({'subgroup': [unprivileged],
                                    'independence': [r_ind],
                                    'separation': [r_sep],
                                    'sufficiency': [r_suf]})
            

            data = pd.concat([data, to_append])

        to_append = pd.DataFrame({'subgroup': [privileged],
                                'independence': [1],
                                'separation': [1],
                                'sufficiency': [1]})

        data.index = data.subgroup
        data = data.iloc[:, 1:]
        data_list.append(data)
    return data_list, l




def kendall_correlation_with_significance(df_list_ind):
    # Calculate correlations
    df1 = df_list_ind[0].corr(method='kendall')
    df2 = df_list_ind[1].corr(method='kendall')
    df3 = df_list_ind[2].corr(method='kendall')
    
    # Function to calculate p-value for Kendall correlation
    def kendall_pvalue(x, y):
        tau, p_value = stats.kendalltau(x, y)
        return p_value
    
    # Calculate p-values for each dataset
    p_values1 = pd.DataFrame(
        [[kendall_pvalue(df_list_ind[0][c1], df_list_ind[0][c2]) for c2 in df1.columns] for c1 in df1.index],
        index=df1.index, columns=df1.columns
    )
    
    p_values2 = pd.DataFrame(
        [[kendall_pvalue(df_list_ind[1][c1], df_list_ind[1][c2]) for c2 in df2.columns] for c1 in df2.index],
        index=df2.index, columns=df2.columns
    )
    
    p_values3 = pd.DataFrame(
        [[kendall_pvalue(df_list_ind[2][c1], df_list_ind[2][c2]) for c2 in df3.columns] for c1 in df3.index],
        index=df3.index, columns=df3.columns
    )
    
    # Format correlations with asterisks for significant values
    def format_value(val1, val2, val3, p1, p2, p3):
        sig1 = '*' if p1 < 0.05 else ''
        sig2 = '*' if p2 < 0.05 else ''
        sig3 = '*' if p3 < 0.05 else ''
        return (
            f"{round(val1,2)}{sig1}",
            f"{round(val2,2)}{sig2}",
            f"{round(val3,2)}{sig3}"
        )
    
    # Create merged dataframe with formatted values
    merged_df = pd.DataFrame(
        {
            col: [
                format_value(
                    df1.loc[row, col],
                    df2.loc[row, col],
                    df3.loc[row, col],
                    p_values1.loc[row, col],
                    p_values2.loc[row, col],
                    p_values3.loc[row, col]
                )
                for row in df1.index
            ]
            for col in df1.columns
        },
        index=df1.index
    )
    
    return merged_df

import pandas as pd
import numpy as np
from scipy import stats

def spearman_correlation_with_significance(df_list_ind):
    # Calculate correlations
    df1 = df_list_ind[0].corr(method='spearman')
    df2 = df_list_ind[1].corr(method='spearman')
    df3 = df_list_ind[2].corr(method='spearman')
    
    # Function to calculate p-value for Spearman correlation
    def spearman_pvalue(x, y):
        rho, p_value = stats.spearmanr(x, y)
        return p_value
    
    # Calculate p-values for each dataset
    p_values1 = pd.DataFrame(
        [[spearman_pvalue(df_list_ind[0][c1], df_list_ind[0][c2]) for c2 in df1.columns] for c1 in df1.index],
        index=df1.index, columns=df1.columns
    )
    
    p_values2 = pd.DataFrame(
        [[spearman_pvalue(df_list_ind[1][c1], df_list_ind[1][c2]) for c2 in df2.columns] for c1 in df2.index],
        index=df2.index, columns=df2.columns
    )
    
    p_values3 = pd.DataFrame(
        [[spearman_pvalue(df_list_ind[2][c1], df_list_ind[2][c2]) for c2 in df3.columns] for c1 in df3.index],
        index=df3.index, columns=df3.columns
    )
    
    # Format correlations with asterisks for significant values
    def format_value(val1, val2, val3, p1, p2, p3):
        sig1 = '*' if p1 < 0.05 else ''
        sig2 = '*' if p2 < 0.05 else ''
        sig3 = '*' if p3 < 0.05 else ''
        return (
            f"{round(val1,2)}{sig1}",
            f"{round(val2,2)}{sig2}",
            f"{round(val3,2)}{sig3}"
        )
    
    # Create merged dataframe with formatted values
    merged_df = pd.DataFrame(
        {
            col: [
                format_value(
                    df1.loc[row, col],
                    df2.loc[row, col],
                    df3.loc[row, col],
                    p_values1.loc[row, col],
                    p_values2.loc[row, col],
                    p_values3.loc[row, col]
                )
                for row in df1.index
            ]
            for col in df1.columns
        },
        index=df1.index
    )
    
    return merged_df

