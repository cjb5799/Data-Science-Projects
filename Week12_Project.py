# Week 12 Final project file
#Name Carla Bradley

import pandas
import thinkstats2
import thinkplot
import numpy as np
import statsmodels.formula.api as smf



def read_data():

    # get directory name and read the file
    dirname = '/Users/carla/Desktop/Python/Week 12/'
    df = pandas.read_csv(dirname + "happyreport4Pt.csv")
    print(df)

    row_idx = df['Country'].loc[lambda x: x == 'Peru'].index
    for r_idx in row_idx:
        print(df.loc[[r_idx]])

        # Year
        # Country name
        # Regional indicator
        # Ladder score
        # GDP per capita
        # Social support
        # Healthy life expectancy
        # Freedom to make life choices
        # Generosity
        # Perceptions of corruption
    return df


# Describes the variables used using type(). Should show as a dataframe or Panda series
def var_types(df):

    # Sample data type and definition
    print(type(df['Year']))
    print(type(df['Country']))
    print(type(df['Regional indicator']))
    print(type(df['Ladder score']))
    print(type(df['GDP per capita']))
    print(type(df['Social support']))
    print(type(df['Healthy life expectancy']))
    print(type(df['Freedom to make life choices']))
    print(type(df['Generosity']))
    print(type(df['Perceptions of corruption']))
    print('\n')


def create_subset(df):
    # create a subset of the data for 2019 and 2020
    df_2019 = df[df['Year'] == 2019]
    df_2020 = df[df['Year'] == 2020]
    print(type(df_2019))
    print(type(df_2020))


def find_outliers(df):
    # find the outliers or zero values
    happiness_index_outliers = df[df['Ladder score'] == 0]
    hio = happiness_index_outliers[['Country', 'Ladder score']]

    gdp_outliers = df[df['GDP per capita'] == 0]
    gdpo = gdp_outliers[['Country', 'GDP per capita']]

    soc_supp_outliers = df[df['Social support'] == 0]
    sso = soc_supp_outliers[['Country', 'Social support']]

    healthy_life_outliers = df[df['Healthy life expectancy'] == 0]
    hlo = healthy_life_outliers[['Country', 'Healthy life expectancy']]

    freedom_outliers = df[df['Freedom to make life choices'] == 0]
    fo = freedom_outliers[['Country', 'Freedom to make life choices']]

    generosity_outliers = df[df['Generosity'] == 0]
    geno = generosity_outliers[['Country', 'Generosity']]

    percep_outliers = df[df['Perceptions of corruption'] == 0]
    po = percep_outliers[['Country', 'Perceptions of corruption']]

    print('\nHappiness Index:\n {}\n'
          'GPD:\n {}\n'
          'Social support:\n {}\n'
          'Healthy life expectancy:\n {}\n'
          'Freedom to make life choices:\n {}\n'
          'Generosity:\n {}\n'
          'Perceptions of corruption:\n {}\n'.format(hio, gdpo, sso, hlo, fo, geno, po))


def display_hist(df):
    width = 0.01

    happy_index = df['Ladder score']
    hist = thinkstats2.Hist(happy_index, label='Happiness Index')
    happiness_index_mean = thinkstats2.Mean(df['Ladder score'])
    thinkplot.Hist(hist, width=width)
    thinkplot.Vlines(happiness_index_mean, y1=0, y2=10, color='green', label='Happiness Index Mean')
    thinkplot.Show(xlabel='value', ylabel='frequency')

    gdp = df['GDP per capita']
    hist = thinkstats2.Hist(gdp, label='GDP')
    thinkplot.Hist(hist, width=width)
    thinkplot.Show(xlabel='value', ylabel='frequency')

    social_supp = df['Social support']
    hist = thinkstats2.Hist(social_supp, label='Social support')
    thinkplot.Hist(hist, width=width)
    thinkplot.Show(xlabel='value', ylabel='frequency')

    healthy_life = df['Healthy life expectancy']
    hist = thinkstats2.Hist(healthy_life, label='Healthy life expectancy')
    thinkplot.Hist(hist, width=width)
    thinkplot.Show(xlabel='value', ylabel='frequency')

    freedom = df['Freedom to make life choices']
    hist = thinkstats2.Hist(freedom, label='Freedom to make life choices')
    thinkplot.Hist(hist, width=width)
    thinkplot.Show(xlabel='value', ylabel='frequency')


# Summary Statistics: Mean, Median, Variance, and Standard Deviation
def display_mean(df):
    happiness_index_mean = thinkstats2.Mean(df['Ladder score'])
    happiness_index_median = thinkstats2.Median(df['Ladder score'])
    happiness_index_var = thinkstats2.Var(df['Ladder score'])
    happiness_index_sd = thinkstats2.Std(df['Ladder score'])

    df_group = df.groupby('Year')['Ladder score'].nlargest(5)
    print(df_group)

    print('Happiness Index Mean: {:.3f}'.format(happiness_index_mean))
    print('Happiness Index Median: {:.3f}'.format(happiness_index_median))
    print('Happiness Index Variance: {:.3f}'.format(happiness_index_var))
    print('Happiness Index Standard Deviation: {:.3f}'.format(happiness_index_sd))
    print("\n")

    gdp_mean = thinkstats2.Mean(df['GDP per capita'])
    gdp_median = thinkstats2.Median(df['GDP per capita'])
    gdp_var = thinkstats2.Var(df['GDP per capita'])
    gdp_sd = thinkstats2.Std(df['GDP per capita'])
    print('GDP Mean: {:.3f}'.format(gdp_mean))
    print('GDP Median: {:.3f}'.format(gdp_median))
    print('GDP Variance: {:.3f}'.format(gdp_var))
    print('GDP Standard Deviation: {:.3f}'.format(gdp_sd))
    print("\n")

    social_supp_mean = thinkstats2.Mean(df['Social support'])
    social_supp_median = thinkstats2.Median(df['Social support'])
    social_supp_var = thinkstats2.Var(df['Social support'])
    social_supp_sd = thinkstats2.Std(df['Social support'])
    print('Social support Mean: {:.3f}'.format(social_supp_mean))
    print('Social support Median: {:.3f}'.format(social_supp_median))
    print('Social support Variance: {:.3f}'.format(social_supp_var))
    print('Social support Standard Deviation: {:.3f}'.format(social_supp_sd))
    print("\n")

    healthy_life_mean = thinkstats2.Mean(df['Healthy life expectancy'])
    healthy_life_median = thinkstats2.Median(df['Healthy life expectancy'])
    healthy_life_var = thinkstats2.Var(df['Healthy life expectancy'])
    healthy_life_sd = thinkstats2.Std(df['Healthy life expectancy'])
    print('Healthy life expectancy Mean: {:.3f}'.format(healthy_life_mean))
    print('Healthy life expectancy Median: {:.3f}'.format(healthy_life_median))
    print('Healthy life expectancy Variance: {:.3f}'.format(healthy_life_var))
    print('Healthy life expectancy Standard Deviation: {:.3f}'.format(healthy_life_sd))
    print("\n")

    freedom_mean = thinkstats2.Mean(df['Freedom to make life choices'])
    freedom_median = thinkstats2.Median(df['Freedom to make life choices'])
    freedom_var = thinkstats2.Var(df['Freedom to make life choices'])
    freedom_sd = thinkstats2.Std(df['Freedom to make life choices'])
    print('Freedom to make life choices Mean: {:.3f}'.format(freedom_mean))
    print('Freedom to make life choices Median: {:.3f}'.format(freedom_median))
    print('Freedom to make life choices Variance: {:.3f}'.format(freedom_var))
    print('Freedom to make life choices Standard Deviation: {:.3f}'.format(freedom_sd))
    print("\n")


# Probability variables for re-use
def prob_vars(df):
    happiness_top5 = df['Ladder score'].sort_values(ascending=False).head(n=5)
    happiness_allother = df['Ladder score'].sort_values(ascending=False).tail(n=299)

    return happiness_top5, happiness_allother


# Probability Mass Function and histogram plot of the Happiness Index
def display_pmf(df):
    happiness_top5 = df['Ladder score'].sort_values(ascending=False).head(n=5)
    happiness_allother = df['Ladder score'].sort_values(ascending=False).tail(n=299)
    ind_count = df.count()
    print(happiness_top5)
    print(ind_count)
    print(happiness_allother)

    pmf_top5 = thinkstats2.Pmf(happiness_top5)
    pmf_allother = thinkstats2.Pmf(happiness_allother)
    print('pmftop5', pmf_top5)
    print('pmfallother', pmf_allother)

    # pmf = thinkstats2.Pmf([1, 1, 2, 3, 4, 4, 5])
    # print('pmf', pmf)
    width = .02

    thinkplot.PrePlot(2, cols=2)
    thinkplot.Hist(pmf_top5, align='right', width=width)
    thinkplot.Hist(pmf_allother, align='left', width=width)
    thinkplot.Config(xlabel='happiness index',
                     ylabel='probability')

    thinkplot.PrePlot(2)
    thinkplot.SubPlot(2)
    thinkplot.Pmfs([pmf_top5, pmf_allother])
    thinkplot.Show(xlabel='Happiness Index')

    return happiness_top5, happiness_allother


# Cumulative Distribution Function and CDF plot of the Happiness Index
def display_cdf(df, p_vars):

    cdf_happy_top5, cdf_all_other = p_vars
    happiness_index = df['Ladder score']
    cdf = thinkstats2.Cdf(happiness_index, label='Happiness Index')

    thinkplot.Cdf(cdf)
    thinkplot.Show(xlabel='Happiness Index Value', ylabel='CDF')
    print(cdf)
    print(cdf_happy_top5)


def display_analytical_dist(df):

    # normal probability plot
    happiness_index = df['Ladder score']
    happiness_index_mean = thinkstats2.Mean(happiness_index)
    happiness_index_sd = thinkstats2.Std(happiness_index)

    # random sample
    mu, sigma = 0, 0.1
    random_sample = np.random.normal(mu, sigma, 304)

    # happiness index data for the plot
    xs = [-3, 3]
    fxs, fys = thinkstats2.FitLine(xs, inter=happiness_index_mean, slope=happiness_index_sd)
    thinkplot.Plot(fxs, fys, color='gray', label='Normal Probability Plot model')
    xs, ys = thinkstats2.NormalProbability(happiness_index)
    thinkplot.Plot(xs, ys, label='Happiness Index Scores')

    # random data for the plot
    npxs, npys = thinkstats2.NormalProbability(random_sample)
    thinkplot.Plot(npxs, npys, label='Random sample data')

    thinkplot.Show(xlabel='Happiness Index Scores')


def display_scatterplots(df):
    # create scatterplots, covariance, Pearson's correlation

    # Calculate Pearson's Median Skewness to get the value of the skewness of the Happiness Index
    df_sorted = df['Ladder score'].sort_values(ascending=False)
    pearson_ms = thinkstats2.PearsonMedianSkewness(df_sorted)
    print("Pearson's Median Skewness (of Happiness Index):", pearson_ms)

    happiness_index_mean = thinkstats2.Mean(df['Ladder score'])
    happiness_index_median = thinkstats2.Median(df['Ladder score'])

    pdf = thinkstats2.EstimatedPdf(df_sorted)
    thinkplot.Pdf(pdf, label='happiness index')
    thinkplot.Vlines(happiness_index_mean, y1=0, y2=1, color='green',
                     label='Happiness Index Mean - {:.3f}'.format(happiness_index_mean))
    thinkplot.Vlines(happiness_index_median, y1=0, y2=1, color='red',
                     label='Happiness Index Median - {:.3f}'.format(happiness_index_median))
    thinkplot.Show(xlabel='Happiness Index', ylabel='PDF')

    # display scatterplots - Freedom to make life choices and Perceptions of corruption
    freedom_choices = df['Freedom to make life choices']
    perceptions_corrupt = df['Perceptions of corruption']

    # Prepare for plots next to each other

    # thinkplot.PrePlot(2, cols=2)  # prepare plots next to each other
    thinkplot.Scatter(freedom_choices, perceptions_corrupt)
    # thinkplot.Config(xlabel='Freedom to make life choices',
    #                ylabel='Perceptions of corruption',
    #                axis=[0, 1, 0, 1])  # Use config when putting the plots next to each other
    thinkplot.Show(xlabel='Freedom to make life choices',
                   ylabel='Perceptions of corruption',
                   axis=[0, 1, 0, 1])

    print('\nFreedom to make life choices: min {:.3f}, max {:.3f}'.format(min(freedom_choices), max(freedom_choices)))
    print('Perceptions of corruption: min {:.3f}, max {:.3f}'.format(min(perceptions_corrupt), max(perceptions_corrupt)))

    # display scatterplots - Social support and Healthy life expectancy
    social_support = df['Social support']
    healthy_life = df['Healthy life expectancy']

    # thinkplot.PrePlot(2)  # prepare plots next to each other
    # thinkplot.SubPlot(2)  # prepare plots next to each other
    thinkplot.Scatter(social_support, healthy_life)
    thinkplot.Show(xlabel='Social support',
                   ylabel='Healthy life expectancy',
                   axis=[0, 2, 0, 2])

    print('\nSocial support: min {:.3f}, max {:.3f}'.format(min(social_support), max(social_support)))
    print('Healthy life expectancy: min {:.3f}, max {:.3f}'.format(min(healthy_life), max(healthy_life)))


def hypothesis_testing(df):

    # These are the data variables for doing the CorrelationPermute test for hypothesis testing
    social_support = df['Social support']
    healthy_life_expectancy = df['Healthy life expectancy']
    data = social_support, healthy_life_expectancy
    ht = CorrelationPermute(data)
    pvalue = ht.PValue()
    print('P-value:', pvalue)
    return pvalue

# This class is to do the CorrelationPermutation between the Social support and Healthy life expectancy variables
# which uses a Pearson Correlation get the pvalue or result of the HypothesisTest class/function.
class CorrelationPermute(thinkstats2.HypothesisTest):
    def TestStatistic(self, data):
        xs, ys = data
        test_stat = abs(thinkstats2.Corr(xs, ys))
        return test_stat

    def RunModel(self):
        xs, ys = self.data
        xs = np.random.permutation(xs)
        return xs, ys
#
#
#     live, firsts, others = first.MakeFrames()
#     live = live.dropna(subset=['agepreg', 'totalwgt_lb'])
#     data = live.agepreg.values, live.totalwgt_lb.values
#     ht = CorrelationPermute(data)
#     pvalue = ht.PValue()

def regression(df, p_vars):
    # Perform a regression analysis
    # Regression analysis on the ladder score/happiness index between 2019 and 2020
    # Comparing the 2 values between 2019 and 2020 to see if there is a significant influence with the variables

    # hap_ind_top5, hap_ind_allothers = p_vars
    hap_ind_2019 = df[df['Year'] == 2019]
    hap_ind_2020 = df[df['Year'] == 2020]
    hap_ind_2019_var = hap_ind_2019['Ladder score']
    hap_ind_2020_var = hap_ind_2020['Ladder score']
    formula = 'hap_ind_2019_var ~ hap_ind_2020_var'
    model = smf.ols(formula, data=df)
    results = model.fit()
    print(results)


def test_least_sq(df):
    # An example of doing the Least Squares test to determine the slope and intercept between the 2 variables - Ch 10
    hap_ind_2019 = df[df['Year'] == 2019]
    hap_ind_2020 = df[df['Year'] == 2020]
    hap_ind_2019_var = hap_ind_2019['Ladder score']
    hap_ind_2020_var = hap_ind_2020['Ladder score']

    print(hap_ind_2019_var)
    print(hap_ind_2020_var)

    subset19 = df[['Social support', 'Healthy life expectancy']]
    soc_sup = subset19['Social support']
    healthy_life = subset19['Healthy life expectancy']
    inter, slope = thinkstats2.LeastSquares(soc_sup, healthy_life)
    fit_xs, fit_ys = thinkstats2.FitLine(soc_sup, inter, slope)

    print('fit_xs: {}, fit_ys: {}'.format(fit_xs, fit_ys))
    print('slope: {}, inter: {}'.format(slope, inter))

    thinkplot.Scatter(soc_sup, healthy_life)
    thinkplot.Plot(fit_xs, fit_ys, color='blue', label='Least Squares model')
    thinkplot.Show(xlabel='Social support',
                   ylabel='Healthy life expectancy',
                   axis=[0, 2, 0, 2])


def main():

    global data
    pandas.set_option('display.max_columns', None)

    data = read_data()
    create_subset(data)
    p_vars = prob_vars(data)
    var_types(data)
    find_outliers(data)
    display_hist(data)
    display_mean(data)
    display_pmf(data)
    display_cdf(data, p_vars)
    display_analytical_dist(data)
    display_scatterplots(data)
    testing_data = hypothesis_testing(data)
    regression(data, p_vars)
    test_least_sq(data)




 #Start program
main()
