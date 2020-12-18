import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score
import scipy.stats as ss
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import random 
import statsmodels.tsa.stattools as ts
import warnings
from arch.unitroot import PhillipsPerron, VarianceRatio
import statsmodels.api as sm
from mlfinlab.labeling.trend_scanning import trend_scanning_labels
from collections import defaultdict
import yfinance as yf
import matplotlib.dates as mdates
import datetime

def convertTuple(tup): 
    string =  '-'.join(tup) 
    return string

def convertString(string):
    security_list = string.split('-')
    return security_list

### Lopez de Prado Machine Learning for Asset Management

def numBins(nObs,corr=None):
    # Optimal number of bins for discretization 
    if corr is None: # univariate case
        z=(8+324*nObs+12*(36*nObs+729*nObs**2)**.5)**(1/3.)
        b=round(z/6.+2./(3*z)+1./3) 
    else: # bivariate case
        b=round(2**-.5*(1+(1+24*nObs/(1.-corr**2))**.5)**.5)
    return int(b)

def mutualInfo(x,y,norm=False):
    # mutual information
    bXY=numBins(x.shape[0], corr=np.corrcoef(x,y)[0,1])
    cXY=np.histogram2d(x,y,bXY)[0]
    iXY=mutual_info_score(None,None,contingency=cXY)
    if norm:
        hX=ss.entropy(np.histogram(x,bXY)[0]) # marginal 
        hY=ss.entropy(np.histogram(y,bXY)[0]) # marginal 
        iXY/=min(hX,hY) # normalized mutual information
    return iXY

def calculate_mutual_information(pairs, pairs_list, prices):
    mutual_info_list = []
    for i in range(len(pairs)):
        security_0 = prices[pairs[i][0]]
        security_1 = prices[pairs[i][1]]
        temp = mutualInfo(security_0, security_1, True)
        mutual_info_list.append(temp)
        
    mutual_info_df = pd.DataFrame({'mutual_information':mutual_info_list},
                                  index=pairs_list)
    mutual_info_df.sort_values(by='mutual_information')

def plot_potential_pairs_hist(mutual_info_df, potential_pairs):
    figsize=(10, 5)
    fontsize=14
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title('All Pairs Training Sample', fontsize=fontsize)
    bins = np.linspace(0, 1, 100)
    ax.hist(mutual_info_df.values, bins, color='#1f77b4', alpha=0.5)
    ax.hist(potential_pairs.values, bins, color='#1f77b4')
    ax.axvline(min(potential_pairs.values), color='red', ls='--')
    ax.legend(['Cutoff', 'Excluded Pairs', 'Potential Pairs'])
    ax.set_xlabel('Mutual Information Score', fontsize=fontsize)
    ax.set_ylabel('Count', fontsize=fontsize)
    fig.tight_layout()

# https://austinrochford.com/posts/2013-12-12-polynomial-regression-cross-validation.html

class PolynomialRegression(BaseEstimator):
    def __init__(self, deg=None):
        self.deg = deg
    
    def fit(self, X, y, deg=None):
        self.model = LinearRegression(fit_intercept=False)
        self.model.fit(np.vander(X, N=self.deg + 1), y)
    
    def predict(self, x):
        return self.model.predict(np.vander(x, N=self.deg + 1))
    
    @property
    def coef_(self):
        return self.model.coef_

def generate_cv_dataframes(pairs, pairs_index, train, test, degrees=np.arange(1, 5), cv=5):
    best_params = []
    best_estimator = []
    best_predicted = []
    best_spread = []
    best_spread_test = []

    for i in range(len(pairs)):
        security_x = train[pairs[i][0]]
        security_x_test = test[pairs[i][0]]
        security_y = train[pairs[i][1]]
        security_y_test = test[pairs[i][1]]

        estimator = PolynomialRegression()
        degrees = degrees
        cv_model = GridSearchCV(estimator,
                                param_grid={'deg': degrees},
                                scoring='neg_mean_squared_error',
                                cv=cv)
        temp_model = cv_model.fit(security_x, security_y)
        temp_params = temp_model.best_params_['deg']
        temp_estimator = temp_model.best_estimator_.coef_
        temp_predicted = temp_model.predict(security_x)
        temp_predicted_test = temp_model.predict(security_x_test)
        temp_spread = (security_y - temp_predicted).values
        temp_spread_test = (security_y_test - temp_predicted_test).values

        best_params.append(temp_params)
        best_estimator.append(temp_estimator)
        best_predicted.append(temp_predicted)
        best_spread.append(temp_spread)
        best_spread_test.append(temp_spread_test)

    cv_predicted = np.array(best_predicted).T
    print(cv_predicted.shape)
    cv_predicted = pd.DataFrame(cv_predicted,
                                columns=pairs_index,
                                index=train.index)
    
    cv_models = pd.DataFrame({'deg':best_params,
                              'estimates':best_estimator},
                             index=pairs_index)

    cv_spreads_train = np.array(best_spread).T
    cv_spreads_train = pd.DataFrame(cv_spreads_train, 
                                    columns=pairs_index,
                                    index=train.index)

    cv_spreads_test = np.array(best_spread_test).T
    cv_spreads_test = pd.DataFrame(cv_spreads_test,
                                   columns=pairs_index,
                                   index=test.index)
    
    return cv_predicted, cv_models, cv_spreads_train, cv_spreads_test

def plot_cv_model_degrees(cv_models):
    figsize = (10, 5)
    fontsize = 14
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title('Potential Pair Corss-Validated Polynomial Regressions', fontsize=fontsize)
    unique_degrees = np.sort(cv_models['deg'].unique())
    degree_count = cv_models['deg'].value_counts()
    ax.bar(x=unique_degrees, height=degree_count)
    ax.set_xlabel('Number of Degrees', fontsize=fontsize)
    ax.set_ylabel('Count', fontsize=fontsize)
    ax.set_xticks(unique_degrees)
    fig.tight_layout()

def hurst(norm_spread):
    """
    Calculates Hurst exponent.
    https://en.wikipedia.org/wiki/Hurst_exponent

    :param norm_spread: An array like object used to calculate half-life.
    """
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    diffs = [np.subtract(norm_spread[l:], norm_spread[:-l]) for l in lags]
    tau = [np.sqrt(np.std(diff)) for diff in diffs]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    H = poly[0]*2.0

    return H

def half_life(norm_spread):
    """
    Calculates time series half-life.
    https://en.wikipedia.org/wiki/Half-life

    :param norm_spread: An array like object used to calculate half-life.
    """
    lag = norm_spread.shift(1)
    lag[0] = lag[1]

    ret = norm_spread - lag
    lag = sm.add_constant(lag)

    model = sm.OLS(ret, lag)
    result = model.fit()
    half_life = -np.log(2)/result.params.iloc[1]

    return half_life

def plot_mean_reversion_statistics(cv_hurst_exponents, cv_half_lives):

    warnings.filterwarnings('ignore')
    
    figsize = (20, 5)
    fontsize = 14
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=figsize)
    fig.suptitle('Mean Reversion Statistics', fontsize=20)
    axs[0].set_title('Hurst Exponent', fontsize=fontsize)
    axs[0].hist(cv_hurst_exponents.values)
    axs[0].set_ylabel('Count', fontsize=fontsize)
    axs[0].set_xlabel('Value', fontsize=fontsize)
    
    clipped_half_lives = np.clip(cv_half_lives.values, 1, 252)
    xlabels = ['', '1-', '50', '100', '150', '200', '250+']
    axs[1].set_title('Half-Life', fontsize=fontsize)
    axs[1].hist(clipped_half_lives)
    axs[1].set_xlabel('Value', fontsize=fontsize)
    axs[1].set_xticklabels(xlabels)
    fig.show()

def generate_stationarity_dataframe(potential_pairs_index, cv_spreads_train):
    
    adfuller_t = []
    adfuller_p = []
    kpss_t = []
    kpss_p = []
    pp_t = []
    pp_p = []
    vr_t = []
    vr_p = []
    
    warnings.filterwarnings('ignore')
    for i, pair in enumerate(potential_pairs_index):
        temp_spread = cv_spreads_train[pair]

        temp_adfuller = ts.adfuller(temp_spread)
        temp_adfuller_t = temp_adfuller[0]
        temp_adfuller_p = temp_adfuller[1]

        temp_kpss = ts.kpss(temp_spread)
        temp_kpss_t = temp_kpss[0]
        temp_kpss_p = temp_kpss[1]

        temp_pp = PhillipsPerron(temp_spread)
        temp_pp_t = temp_pp.stat
        temp_pp_p = temp_pp.pvalue

        temp_vr = VarianceRatio(temp_spread)
        temp_vr_t = temp_vr.stat
        temp_vr_p = temp_vr.pvalue

        adfuller_t.append(temp_adfuller_t)
        adfuller_p.append(temp_adfuller_p)

        kpss_t.append(temp_kpss_t)
        kpss_p.append(temp_kpss_p)

        pp_t.append(temp_pp_t)
        pp_p.append(temp_pp_p)

        vr_t.append(temp_vr_t)
        vr_p.append(temp_vr_p)

    cv_stationary_tests = pd.DataFrame({'adf_t_stat':adfuller_t,
                                        'adf_p_value':adfuller_p,
                                        'kpss_t_stat':kpss_t,
                                        'kpss_p_value':kpss_p,
                                        'pp_t_stat':pp_t,
                                        'pp_p_value':pp_p,
                                        'vr_t_stat':vr_t,
                                        'vr_p_value':vr_p},
                                       index=potential_pairs_index)
    return cv_stationary_tests


def plot_spreads(cv_filtered, cv_predicted, cv_spreads, prices):
    
    for pair in cv_filtered:
        fontsize=14
        securities = convertString(pair)

        fig, axs = plt.subplots(3, 1, sharex=False, figsize=(20, 10))
        security = securities[0]
        color = 'tab:blue'
        axs[0].plot(prices[security].values, color=color)
        axs[0].set_ylabel(security, color=color, fontsize=fontsize)
        axs[0].tick_params(axis='y', labelcolor=color)
        axs[0].set_title(f'{pair}', fontsize=fontsize)

        security = securities[1]
        color = 'tab:orange'
        axs2 = axs[0].twinx()
        axs2.plot(prices[security].values, color=color)
        axs2.set_ylabel(security, color=color, fontsize=fontsize)
        axs2.tick_params(axis='y', labelcolor=color)
        axs2.set_xlabel('Date Index', fontsize=fontsize)

        axs[1].plot(prices[securities[0]].values, cv_predicted[convertTuple(securities)].values, 'r')
        axs[1].scatter(prices[securities[0]].values, prices[securities[1]].values, alpha=0.1)
        axs[1].set_xlabel(f'{securities[0]} Price', fontsize=fontsize)
        axs[1].set_ylabel(f'{securities[1]} Price', fontsize=fontsize)

        axs[2].plot(cv_spreads[convertTuple(securities)].values)
        axs[2].set_xlabel('Date Index', fontsize=fontsize)
        axs[2].set_ylabel('Spread', fontsize=fontsize)

        fig.tight_layout()

def generate_trend_labels(securities, spreads):
    
    trend = trend_scanning_labels(spreads[securities[0]], look_forward=False)
    trend_t1 = trend['t1'].values.T
    trend_t_value = trend['t_value'].values.T
    trend_ret = trend['ret'].values.T
    trend_bin = trend['bin'].values.T
    
    for i in range(1, len(securities)):
        temp = trend = trend_scanning_labels(spreads[securities[i]], look_forward=False)
        
        temp_t1 = temp['t1'].values.T
        temp_t_value = temp['t_value'].values.T
        temp_ret = temp['ret'].values.T
        temp_bin = temp['bin'].values.T
        
        trend_t1 = np.hstack([trend_t1, temp_t1])
        trend_t_value = np.hstack([trend_t_value, temp_t_value])
        trend_ret = np.hstack([trend_ret, temp_ret])
        trend_bin = np.hstack([trend_bin, temp_bin])
        
    dates = spreads.index
    index = pd.MultiIndex.from_product([securities, dates],
                                      names=['pair', 'date'])
    trend_labels = pd.DataFrame({'t1':trend_t1,
                                   't_value':trend_t_value,
                                   'ret':trend_ret,
                                   'bin':trend_bin},
                                  index=index)
    trend_labels = trend_labels.swaplevel(0,1).sort_index(0) 
    return trend_labels

def plot_trend_labels(filtered_pairs, spreads, trend_labels):
    idx = pd.IndexSlice

    for pair in filtered_pairs:
        x = np.array(range(len(spreads.dropna())))
        y = np.array(spreads[pair].dropna())
        group = np.array(trend_labels.loc[idx[:, pair], 'bin'].dropna())
        cdict = {1.0:'green', -1.0:'red'}
        
        fontsize = 14
        figsize = (20, 5)
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(f'{pair} Spread', fontsize=fontsize)
        for g in np.unique(group):
            ix = np.where(group == g)
            ax.scatter(x[ix], y[ix], c=cdict[g], s=10)
        ax.legend(['short spread', 'long spread'], fontsize=fontsize)
        ax.set_ylabel('Spread ($)', fontsize=fontsize)
        ax.set_xlabel('Date Index', fontsize=fontsize)
        fig.tight_layout()

def get_prices_and_states(date, pair, bin_df, prices):
    idx = pd.IndexSlice
    state_0 = bin_df.loc[idx[:, pair], 'bin'].shift(1)[date].values[0]
    state_1 = bin_df.loc[idx[date, pair], 'bin']
    securities = convertString(pair)
    security_0 = securities[0]
    security_1 = securities[1]
    
    price_0 = prices.loc[date, security_0]
    price_1 = prices.loc[date, security_1]
    return price_0, price_1, state_0, state_1

def create_spread_portfolio(date, bin_df,
                            pair, wealth,
                            leverage_pct, daily_interest_rate,
                            maintenance_pct, slippage_pct,
                            prices):
    
    # Start with initial amount of cash in hand
    cash = wealth
    leveraged_cash = cash*(1+leverage_pct) # Leverage cash
    loan = cash*leverage_pct
    short_leg_value = leveraged_cash # Dollar amount of short leg
    long_leg_value = short_leg_value # Dollar amount of long leg
 
    # Get prices and states to create portfolio
    price_0, price_1, state_0, state_1 = get_prices_and_states(date, pair, bin_df, prices)
    
    if state_1 == 1.0:
        price_long = price_0
        price_short = price_1
        
    else:
        price_long = price_1
        price_short = price_0
    
    # Assume strategy does not get best price
    price_long_slip = price_long*(1+slippage_pct)
    price_short_slip = price_short*(1+slippage_pct)
    
    shares_long = long_leg_value/price_long_slip
    shares_short = short_leg_value/price_short_slip
    
    long_leg = shares_long*price_long
    short_leg = shares_short*price_short
    margin_account = long_leg + cash 
    maintenance_req = maintenance_pct*short_leg
    margin_call = 0.0
    
    portfolio = long_leg - short_leg
    wealth = portfolio + cash

    portfolio_dict = {'wealth':wealth,
                      'portfolio':portfolio,
                      'cash':cash,
                      'long_leg':long_leg,
                      'short_leg':short_leg,
                      'shares_long':shares_long,
                      'shares_short':shares_short,
                      'loan':loan,
                      'daily_interest_rate':daily_interest_rate,
                      'margin_account':margin_account,
                      'maintenance_req':maintenance_req,
                      'maintenance_pct':maintenance_pct,
                      'leverage_pct':leverage_pct,
                      'margin_call':margin_call,
                      'slippage_pct':slippage_pct,
                      'pair':pair,
                      'state_0':state_0,
                      'state_1':state_1}
    return portfolio_dict

def step_portfolio(date, bin_df, portfolio_dict, prices,
                   leverage_pct, daily_interest_rate,
                   maintenance_pct, slippage_pct):
    
    step_list = ['wealth', 'portfolio',
                 'long_leg', 'short_leg',
                 'shares_long', 'shares_short',
                 'maintenance_req', 'margin_account',
                 'margin_call', 'cash', 'loan',
                 'state_0', 'state_1']
    wealth = portfolio_dict['wealth']
    if wealth == 0.0:
        for item in step_list:
            portfolio_dict[item] = 0.0
        pre_margin_portfolio = portfolio_dict
        return pre_margin_portfolio
    else:
        # Get updated prices and state from portfolio dictionary
        pair = portfolio_dict['pair']
        price_0, price_1, state_0, state_1 = get_prices_and_states(date, pair, bin_df, prices)

        if np.isnan(state_0):
            state_0 = state_1

        elif state_0 == 1.0:
            price_long = price_0
            price_short = price_1

        else:
            price_long = price_1
            price_short = price_0

        # Calculate portfolio leg values
        shares_long = portfolio_dict['shares_long']
        shares_short = portfolio_dict['shares_short']
        long_leg = shares_long*price_long
        short_leg = shares_short*price_short

        # Calculate total margin required and margin call
        maintenance_pct = portfolio_dict['maintenance_pct']
        maintenance_req = maintenance_pct*short_leg
        cash = portfolio_dict['cash']
        margin_account = long_leg + cash
        margin_call = max(0.0, maintenance_req - margin_account)

        # Calculate portfolio value
        portfolio = long_leg - short_leg
        loan = portfolio_dict['loan']
        daily_interest_rate = portfolio_dict['daily_interest_rate']
        interest = loan * daily_interest_rate
        cash = portfolio_dict['cash']
        cash = cash - interest
        wealth = max(0.0, portfolio + cash)

        # Update portfolio dictionary
        for item in step_list:
            portfolio_dict[item] = eval(item)

        pre_margin_portfolio = portfolio_dict
        return pre_margin_portfolio

def margin_call_rebalance(date, bin_df, pre_margin_portfolio, prices,
                          leverage_pct, daily_interest_rate,
                          maintenance_pct, slippage_pct):

    # Reduce wealth
    margin_call = pre_margin_portfolio['margin_call']
    wealth = pre_margin_portfolio['wealth']
    wealth = wealth - margin_call

    # Load additional data to create new spread portfolio
    pair = pre_margin_portfolio['pair']
    slippage_pct = pre_margin_portfolio['slippage_pct']
    
    # Create spread portfolio
    portfolio_dict = create_spread_portfolio(date, bin_df,
                                             pair, wealth,
                                             leverage_pct, daily_interest_rate,
                                             maintenance_pct, slippage_pct,
                                             prices)
   # print('margin rebalance')
    return portfolio_dict

def step_and_rebalance_portfolio(date, bin_df, portfolio_dict, prices,
                                 leverage_pct, daily_interest_rate,
                                 maintenance_pct, slippage_pct):
    
    # Load pre-step portfolio information and step portfolio
    pair = portfolio_dict['pair']
    pre_margin_portfolio = step_portfolio(date, bin_df, portfolio_dict, prices,
                                          leverage_pct, daily_interest_rate,
                                          maintenance_pct, slippage_pct)
    
    # Check if portfolio requires a margin call
    margin_call = pre_margin_portfolio['margin_call']
    if margin_call > 0.0:
        # Margin call required
      #  print('margin call required')
        # Reduce wealth by and rebalance portfolio
        portfolio_dict = margin_call_rebalance(date, bin_df, pre_margin_portfolio, prices,
                                               leverage_pct, daily_interest_rate,
                                               maintenance_pct, slippage_pct)
    else:
        # No margin call required
        post_margin_portfolio = pre_margin_portfolio
        state_0 = post_margin_portfolio['state_0']
        state_1 = post_margin_portfolio['state_1']
        if state_0 == state_1:
            portfolio_dict = post_margin_portfolio
        else:
            pair = post_margin_portfolio['pair']
            wealth = post_margin_portfolio['wealth']
            leverage_pct = post_margin_portfolio['leverage_pct']
            maintenance_pct = post_margin_portfolio['maintenance_pct']
            slippage_pct = post_margin_portfolio['slippage_pct']
            portfolio_dict = create_spread_portfolio(date, bin_df,
                                                     pair, wealth,
                                                     leverage_pct, daily_interest_rate,
                                                     maintenance_pct, slippage_pct,
                                                     prices)
    return portfolio_dict

def calculate_portfolio_df(pairs, bin_df, prices,
                           leverage_pct=0.0, daily_interest_rate=0.0011,
                           maintenance_pct=0.4, slippage_pct=0.0001):
    dates = bin_df.index.get_level_values(0).unique()
    portfolio_dict_list = []
    for p, pair in enumerate(pairs):
        temp_portoflio_dict = defaultdict(list)
        for i, date in enumerate(dates):
            if i == 0:
                wealth_0 = 1/len(pairs)
                portfolio_dict = create_spread_portfolio(dates[i], bin_df,
                                                         pair, wealth_0,
                                                         leverage_pct, daily_interest_rate,
                                                         maintenance_pct, slippage_pct,
                                                         prices)
                for k, v in portfolio_dict.items():
                    temp_portoflio_dict[k].append(v)
            else:
                portfolio_dict = step_and_rebalance_portfolio(dates[i], bin_df, portfolio_dict, prices,
                                                              leverage_pct, daily_interest_rate,
                                                              maintenance_pct, slippage_pct)
                for k, v in portfolio_dict.items():
                    temp_portoflio_dict[k].append(v)
        portfolio_dict_list.append(temp_portoflio_dict)

    explode_index = pd.MultiIndex.from_product([pairs, dates], names=['pair', 'date']).swaplevel()
    portfolio_df = pd.json_normalize(portfolio_dict_list).apply(pd.Series.explode)
    portfolio_df.index = explode_index
    portfolio_df = portfolio_df.sort_index(0)
    return portfolio_df

def calculate_Sharpe(wealth, N=252):
    sampleMuDaily   = pd.Series(wealth).pct_change().mean()
    sampleMuAnnual  = (sampleMuDaily + 1)**N - 1
    sampleSigDaily  = pd.Series(wealth).pct_change().std()
    sampleVarAnnual = (sampleSigDaily**2 + (sampleMuDaily+1)**2)**N - (sampleMuDaily+1)**(2*N)
    sampleSigAnnual = np.sqrt(sampleVarAnnual)
    
    Sharpe = sampleMuAnnual/sampleSigAnnual
    return Sharpe

def calculate_drawdown(wealth):
    relativeDrawdown = []
    highWater        = wealth[0]
    for i in range(len(wealth)):
        if wealth[i] > highWater:
            highWater = wealth[i]
            relativeDrawdown.append(0)
        else:
            relativeDrawdown.append(wealth[i]/highWater -1)
    relativeDrawdown = pd.Series(relativeDrawdown, index=wealth.index)
    return relativeDrawdown

def calculate_max_drawdown(wealth):
    relativeDrawdown = []
    highWater        = wealth[0]
    for i in range(len(wealth)):
        if wealth[i] > highWater:
            highWater = wealth[i]
            relativeDrawdown.append(0)
        else:
            relativeDrawdown.append(wealth[i]/highWater -1)
    relativeDrawdown = pd.Series(relativeDrawdown, index=wealth.index)
    max_drawdown = relativeDrawdown.min()
    return max_drawdown

def plot_portfolio_performance_summary(portfolio_df, benchmark):
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(20, 10))
    fontsize=14

    total_wealth = portfolio_df['wealth'].unstack().sum(axis=1)
    standardized_benchmark = benchmark/benchmark[0]
    axs[0].set_title('Total Return', fontsize=fontsize)
    axs[0].plot(total_wealth.values)
    axs[0].plot(standardized_benchmark.values, color='red')
    axs[0].set_ylabel('Wealth ($)', fontsize=fontsize)
    axs[0].legend(['Entropy Pairs', 'Benchmark'])

    daily_returns = total_wealth.pct_change()*100
    axs[1].set_title('Daily Return', fontsize=fontsize)
    axs[1].plot(daily_returns.values)
    axs[1].set_ylabel('Percent (%)', fontsize=fontsize)

    # TODO: fix dates for bottom chart
    relative_drawdown = calculate_drawdown(total_wealth)*100
    relative_drawdown_bench = calculate_drawdown(benchmark)*100
    # dates = [datetime.datetime.strptime(d,"%Y-%M-%d").date() for d in relative_drawdown.index]
    # formatter = mdates.DateFormatter("%Y-%M-%d")
    # axs[2].xaxis.set_major_formatter(formatter)
    # locator = mdates.YearLocator()
    # axs[2].xaxis.set_major_locator(locator)
    axs[2].set_title('Relative Drawdown', fontsize=fontsize)
    axs[2].plot(relative_drawdown.values)
    axs[2].plot(relative_drawdown_bench.values, color='r')
    axs[2].set_ylabel('Percent (%)', fontsize=fontsize)
    axs[2].legend(['Entropy Pairs', 'Benchmark'])
    
    net_exposure = portfolio_df['portfolio'].unstack().sum(axis=1)
    net_exposure_pct = net_exposure/total_wealth*100
    average_net_exposure_pct = net_exposure_pct.mean()
    axs[3].set_title('Net Market Exposure', fontsize=fontsize)
    axs[3].plot(net_exposure_pct.values)
    axs[3].axhline(average_net_exposure_pct, ls='--', color='r', label='average')
    axs[3].set_ylabel('Percent (%)', fontsize=fontsize)
    axs[3].legend(['Net Exposure', 'Average Exposure'])

    fig.show()

def plot_pairs_hist(portfolio_df):
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(15, 10))
    fontsize = 14
    bins = 20
    total_wealth = portfolio_df['wealth'].unstack().sum(axis=1)
    daily_returns = total_wealth.pct_change()*100
    mean = daily_returns.mean()
    axs[0].set_title('Daily', fontsize=fontsize)
    axs[0].hist(daily_returns, bins=bins)
    axs[0].axvline(mean, ls='--', color='r')
    axs[0].set_ylabel('Count', fontsize=fontsize)
    axs[0].legend([f'Mean = {round(mean, 2)}%'], fontsize=fontsize)

    weekly_returns = total_wealth.pct_change(5)*100
    mean = weekly_returns.mean()
    axs[1].set_title('Weekly', fontsize=fontsize)
    axs[1].hist(weekly_returns, bins=bins)
    axs[1].axvline(mean, ls='--', color='r')
    axs[1].set_ylabel('Count', fontsize=fontsize)
    axs[1].legend([f'Mean = {round(mean, 2)}%'], fontsize=fontsize)

    monthly_returns = total_wealth.pct_change(20)*100
    mean = monthly_returns.mean()
    axs[2].set_title('Monthly', fontsize=fontsize)
    axs[2].hist(monthly_returns, bins=bins)
    axs[2].axvline(mean, ls='--', color='r')
    axs[2].set_xlabel('Return (%)', fontsize=fontsize)
    axs[2].set_ylabel('Count', fontsize=fontsize)
    axs[2].legend([f'Mean = {round(mean, 2)}%'], fontsize=fontsize)

    fig.show()

def calculate_ann_return(wealth, N=252):
    mean_return   = pd.Series(wealth).pct_change().mean()
    ann_return  = (mean_return + 1)**N - 1
    return ann_return

def calculate_ann_std(wealth, N=252):
    sampleMuDaily   = pd.Series(wealth).pct_change().mean()
    sampleSigDaily  = pd.Series(wealth).pct_change().std()
    sampleVarAnnual = (sampleSigDaily**2 + (sampleMuDaily+1)**2)**N - (sampleMuDaily+1)**(2*N)
    sampleSigAnnual = np.sqrt(sampleVarAnnual)
    return sampleSigAnnual

def plot_pairs_ann_return_hist(portfolio_df):
    fig, axs = plt.subplots(1, 1, figsize=(15, 10))
    fontsize = 14
    bins = np.arange(-1, 4, 0.25)
    dates = portfolio_df.index.get_level_values(0).unique()
    pair_wealth = portfolio_df['wealth'].unstack()
    pair_return = pair_wealth.pct_change()
    pair_mean_return = pair_return.mean()
    pair_ann_return = (1+pair_mean_return)**252 - 1
    median = pair_ann_return.median()
    axs.set_title(f'Pair Returns', fontsize=fontsize)
    axs.hist(np.clip(pair_ann_return, bins[0], bins[-1]), bins=bins)
    axs.axvline(median, ls='--', color='r')
    axs.legend([f'Median = {round(median, 2)*100}%'], fontsize=fontsize)
    xlabels = (bins*100).astype(str)
    xlabels[-2] += '+'
    xlabels[1::2] = ''
    N_labels = len(xlabels)
    axs.set_xlim(xmin=-1.1, xmax=bins[-1])
    axs.set_xticks(bins)
    axs.set_xticklabels(xlabels)
    axs.set_xlabel('Annualized Return (%)', fontsize=fontsize)
    axs.set_ylabel('Count', fontsize=fontsize)
    fig.show()

def calculate_calendar_summary(strategy_list, strategy_labels,
                               statistics_labels, 
                               statistics_functions):
    total_wealth = strategy_list[0]
    total_wealth.index = pd.to_datetime(total_wealth.index)
    calendar_summary_index = pd.MultiIndex.from_product([statistics_labels, strategy_labels],
                                              names=['statistic', 'strategy'])
    calendar_summary_data = []
    for function in statistics_functions:
        for strategy in strategy_list:
            grouper = pd.Grouper(freq='Y')
            calendar_data = strategy.groupby(grouper).apply(function)
            temp_dates = calendar_data.index            
            calendar_values = calendar_data.T.values

            if function != calculate_Sharpe:
                calendar_values *= 100

            calendar_data = [round(value, 2) for value in calendar_values]

            calendar_summary_data.append(calendar_data)

    calendar_summary_df = pd.DataFrame(data=calendar_summary_data,
                                       index=calendar_summary_index,
                                       columns=temp_dates)
    return calendar_summary_df

def calculate_window_summary(strategy_list, strategy_labels,
                             statistics_labels, statistics_functions):
    
    window_summary_index = pd.MultiIndex.from_product([statistics_labels, strategy_labels],
                                                  names=['statistic', 'strategy'])

    window_summary_cols = ['1-Year', '3-Year', '5-Year', '10-Year', 'Since Inception']
    window_summary_data = []
    for function in statistics_functions:
        for strategy in strategy_list:
            window_values = []
            windows = [1, 3, 5, 7, len(strategy)/252]
            for window in windows:
                window_data = strategy.rolling(int(252*window)).apply(function).mean()
                if function != calculate_Sharpe:
                    window_data *= 100
                window_values.append(window_data)
            window_data = [round(value, 2) for value in window_values]
            window_summary_data.append(window_data)
    window_summary_df = pd.DataFrame(data=window_summary_data, 
                                     index=window_summary_index,
                                     columns=window_summary_cols)
    return window_summary_df
