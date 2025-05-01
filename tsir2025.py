import time
import numpy as np
import seaborn as sns
import pandas as pd
import pickle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression

XMM, XSM, XIM, XRM, \
XMS, XSS, XIS, XRS, \
XMI, XSI, XII, XRI, \
XMR, XSR, XIR, XRR, \
VMM, VSM, VIM, VRM, \
VMS, VSS, VIS, VRS, \
VMI, VSI, VII, VRI, \
VMR, VSR, VIR, VRR = range(32)

def read_from_pickle(file_name):
    with open(file_name, "rb") as file:
        items = pickle.load(file)
    return items

def get_parameters_by_names(parameters, names):
    if len(names) == 1:
        return parameters[names[0]]
    return (parameters[name] for name in names)

def age_progression(INP, params):
    ''' advance age; frequency in days; This function advances age PER class, PER age for each tau days.
    It handles fraction populations with a (uniform) chance that one whole individual will advance in age.
    '''
    x, M = np.shape(INP)
    V = INP
    n, tau_age = params  ## n is age_bin_sizes by year (so 1 month = 1/12 = 0.0833)

    rands = np.random.uniform(0, 1, size=(M - 1) * x).reshape((x, M - 1))

    for k in reversed(range(0, M - 1)):  # for each age group
        for i in range(x):  # for each state variable
            move_rate = 1 / (365 * n[k]) * tau_age * V[i, k]
            move_int = np.floor(move_rate)
            ## decide whether or not to move the remainder
            move_frac = move_rate % 1  ## modulus operator; same as np.remainder(move_rate, 1)
            if rands[i, k] < move_frac:
                move_int += 1
            V[i, k + 1] = V[i, k + 1] + move_int
            V[i, k] = V[i, k] - move_int
    return V

def msirv_equations_with_vacc_measles_only(population, parameters, t):
    ''' MSIR model of measles and rubella (no coinfection)
    '''
    exclude_coinfection = True

    population_size_by_age = np.sum(population, axis=0)
    total_population_size = np.sum(population_size_by_age)

    NUMBER_OF_STATES, NUMBER_OF_AGE_GROUPS = np.shape(population)

    ## Add the unborn and dead states
    population = np.vstack((population, np.ones(NUMBER_OF_AGE_GROUPS) * total_population_size))  ## unborn
    population = np.vstack((population, np.zeros(NUMBER_OF_AGE_GROUPS)))  ## dead

    ## Set up states indices + 2 dummy states (X = unvaccinated; V = vaccinated)
    XMM, XSM, XIM, XRM, \
    XMS, XSS, XIS, XRS, \
    XMI, XSI, XII, XRI, \
    XMR, XSR, XIR, XRR, \
    VMM, VSM, VIM, VRM, \
    VMS, VSS, VIS, VRS, \
    VMI, VSI, VII, VRI, \
    VMR, VSR, VIR, VRR, \
    XUNBORN, XDEAD = range(NUMBER_OF_STATES + 2)

    # age_bin_sizes = get_parameters_by_names(parameters, ["age_bin_sizes"])
    mu, birth_rate, fertility_rates_normalized, C = get_parameters_by_names(parameters, ["death_rates", "birth_rate", "fertility_rates_normalized", "contact_matrix"])
    reporting_ratios = get_parameters_by_names(parameters, ["reporting_ratios"])
    mbeta, mdelta, mgamma, msigma = get_parameters_by_names(parameters, ["transmission_rate_m", "importation_rate_m",
                                                                         "recovery_rate_m", "maternal_immunity_waning_rate_m"])
    rbeta, rdelta, rgamma, rsigma = get_parameters_by_names(parameters, ["transmission_rate_r", "importation_rate_r",
                                                                         "recovery_rate_r", "maternal_immunity_waning_rate_r"])
    T, tau, tau_age = get_parameters_by_names(parameters, ["T", "tau", "tau_age"])
    VC, VEM, VER = get_parameters_by_names(parameters, ["vaccine_coverage", "vaccine_eff_m", "vaccine_eff_r"])


    transition_matrix = np.zeros((NUMBER_OF_AGE_GROUPS, NUMBER_OF_STATES + 2, NUMBER_OF_STATES + 2))

    ## Birth rates (with maternal immunity to rubella; measles; and rubella and measles)
    ## states with this type of immunity contributes to the class distribution of new births
    ## in measles only, births would only be XMS or XSS
    xmm_states = [XMM, XRM, XMR, XRR, VMM, VRM, VMR, VRR]
    xsm_states = [XSM, XIM, XSR, XIR, VSM, VIM, VSR, VIR]
    xms_states = [XMS, XRS, XMI, XRI, VMS, VRS, VMI, VRI]
    fertile_population = population[:NUMBER_OF_STATES, :NUMBER_OF_AGE_GROUPS] * fertility_rates_normalized
    # fractions = compute_fractions_born_into_states(population[:NUMBER_OF_STATES, :NUMBER_OF_AGE_GROUPS], xmm_states, xsm_states, xms_states)
    fractions = compute_fractions_born_into_states(fertile_population, xmm_states, xsm_states, xms_states)
    birthRates = birth_rate * total_population_size * fractions

    # Forces of infection
    measles_states = [XIM, XIS, XII, XIR, VIS, VIR, VIM]
    rubella_states = [XMI, XSI, XII, XRI, VSI, VRI, VMI]
    mfoi, rfoi = compute_forces_of_infection(population[:NUMBER_OF_STATES, :NUMBER_OF_AGE_GROUPS], measles_states, rubella_states, t, mbeta, rbeta, C)

    ## Fill in transition matrix
    xbs = [XMS, XSS, XMS, XSS]  # birth states: babies enter population through these states (Changed to measles only, no rubella maternal immunity)
    xma = [XMM, XMS, XMI, XMR, VMM, VMS, VMI, VMR]  # measles: (a)ll that are (m)aternally immune
    xsa = [XSM, XSS, XSI, XSR, VSM, VSS, VSI, VSR]  # measles: (a)ll that are (s)usceptible
    xia = [XIM, XIS, XII, XIR, VIM, VIS, VII, VIR]
    xra = [XRM, XRS, XRI, XRR, VRM, VRS, VRI, VRR]
    #
    # xam = [XMM, XSM, XIM, XRM, VMM, VSM, VIM, VRM]  # rubella: (a)ll that are (m)aternally immune
    # xas = [XMS, XSS, XIS, XRS, VMS, VSS, VIS, VRS]
    # xai = [XMI, XSI, XII, XRI, VMI, VSI, VII, VRI]
    # xar = [XMR, XSR, XIR, XRR, VMR, VSR, VIR, VRR]
    #
    # one_route_vaccinations_from = [XMM, XRM, XMR, XRR] ## these should always be 0 because no hay M and R for rubella
    # one_route_vaccinations_to   = [VMM, VRM, VMR, VRR]

    ## for case tracking:
    transitions_m_cases_unvaccinated = [[XSM, XIM], [XSS, XIS], [XSR, XIR]]
    # transitions_r_cases_unvaccinated = [[XMS, XMI], [XSS, XSI], [XRS, XRI]]
    transitions_m_cases_vaccinated   = [[VSM, VIM], [VSS, VIS], [VSR, VIR]]
    # transitions_r_cases_vaccinated   = [[VMS, VMI], [VSS, VSI], [VRS, VRI]]

    case_count_measles_unvaccinated = np.zeros(NUMBER_OF_AGE_GROUPS)
    case_count_rubella_unvaccinated = np.zeros(NUMBER_OF_AGE_GROUPS)
    case_count_measles_vaccinated = np.zeros(NUMBER_OF_AGE_GROUPS)
    case_count_rubella_vaccinated = np.zeros(NUMBER_OF_AGE_GROUPS)

    ## Births
    for i, j in enumerate(xbs):
        transition_matrix[0, XUNBORN, j] = birthRates[i]

    ## Measles: Maternal immunity waning, transmission and recovery
    for i, j, k, l in zip(xma, xsa, xia, xra):
        transition_matrix[:, i, j] = msigma * population[i]
        transition_matrix[:, j, k] = mfoi   * population[j]
        transition_matrix[:, k, l] = mgamma * population[k]

    # Imported measles cases
    for j in [XIS]:
        transition_matrix[:, XUNBORN, j] = mdelta

    ## Rubella: Maternal immunity waning, transmission and recovery
    # for i, j, k, l in zip(xam, xas, xai, xar):
    #     transition_matrix[:, i, j] = rsigma * population[i]
    #     transition_matrix[:, j, k] = rfoi   * population[j]
    #     transition_matrix[:, k, l] = rgamma * population[k]

    # Imported rubella cases
    # for j in [XSI]:
    #     transition_matrix[:, XUNBORN, j] = rdelta

    if exclude_coinfection:
        # Measles
        transition_matrix[:, XSI, XII] = 0.0
        transition_matrix[:, XII, XRI] = 0.0
        transition_matrix[:, VSI, VII] = 0.0
        transition_matrix[:, VII, VRI] = 0.0
        # Rubella
        transition_matrix[:, XIS, XII] = 0.0
        transition_matrix[:, XII, XIR] = 0.0
        transition_matrix[:, VIS, VII] = 0.0
        transition_matrix[:, VII, VIR] = 0.0

    ## Deaths
    for i in range(NUMBER_OF_STATES):
        transition_matrix[:, i, XDEAD] = mu * population[i]

    ## Routine vaccination
    # 1-possible transition: (XMM, XRM, XMR, XRR) to (VMM, VRM, VMR, VRR)
    # for i, j in zip(one_route_vaccinations_from, one_route_vaccinations_to):  # these should be 0 because no hay rubella M or R class
    #     transition_matrix[:, i, j] = VC * population[i]

    # 2+ possible transitions done manually:
    # i = XSM  # this should also be 0 because no hay rubella maternal immunity
    # transition_matrix[:, i, VRM] = VC * VEM * population[i]         # measles success
    # transition_matrix[:, i, VSM] = VC * (1 - VEM) * population[i]   # measles fail

    i = XMS
    transition_matrix[:, i, VMR] = VC * VER * population[i]         # rubella success
    transition_matrix[:, i, VMS] = VC * (1 - VER) * population[i]   # rubella fail

    i = XSS
    transition_matrix[:, i, VSS] = VC * (1 - VEM) * (1 - VER) * population[i] # measles fail, rubella fail
    transition_matrix[:, i, VRS] = VC * VEM       * (1 - VER) * population[i] # measles success, rubella fail
    # transition_matrix[:, i, VSR] = VC * (1 - VEM) * VER       * population[i] # measles fail, rubella success
    # transition_matrix[:, i, VRR] = VC * VEM       * VER       * population[i] # measles success, rubella success

    i = XRS
    transition_matrix[:, i, VRS] = VC * (1 - VER) * population[i]   # rubella fail
    transition_matrix[:, i, VRR] = VC * VER * population[i]         # rubella success

    # i = XSR
    # transition_matrix[:, i, VSR] = VC * (1 - VEM) * population[i]   # measles fail
    # transition_matrix[:, i, VRR] = VC * VEM * population[i]         # measles success

    ## Compute new population
    for age in range(NUMBER_OF_AGE_GROUPS):
        for i in range(NUMBER_OF_STATES + 2):
            for j in range(NUMBER_OF_STATES + 2):
                transition_rate = transition_matrix[age, i, j]
                if transition_rate > 0.0:
                    number_to_transition = np.random.poisson(transition_rate * tau)
                    if population[i, age] < number_to_transition:
                        number_to_transition = population[i, age]
                    population[i, age] -= number_to_transition
                    population[j, age] += number_to_transition
                    # track cases
                    if [i, j] in transitions_m_cases_unvaccinated:
                        case_count_measles_unvaccinated[age] += number_to_transition
                    # elif [i, j] in transitions_r_cases_unvaccinated:
                    #     case_count_rubella_unvaccinated[age] += number_to_transition
                    elif [i, j] in transitions_m_cases_vaccinated:
                        case_count_measles_vaccinated[age] += number_to_transition
                    # elif [i, j] in transitions_r_cases_vaccinated:
                    #     case_count_rubella_vaccinated[age] += number_to_transition

    population = population[:NUMBER_OF_STATES, :NUMBER_OF_AGE_GROUPS]
    cases = reporting_ratios * np.array([case_count_measles_unvaccinated,
                                         case_count_rubella_unvaccinated,
                                         case_count_measles_vaccinated,
                                         case_count_rubella_vaccinated])
    return population, cases

def compute_fractions_born_into_states(population, xmm_states, xsm_states, xms_states):
    population_size_by_age = np.sum(population, axis=0)  ## length 40
    total_population_size = np.sum(population_size_by_age)

    xmm_populations = [population[s] for s in xmm_states]  ## size (8, 40)
    xsm_populations = [population[s] for s in xsm_states]
    xms_populations = [population[s] for s in xms_states]

    xmm_fraction = np.sum(xmm_populations) / total_population_size
    xsm_fraction = np.sum(xsm_populations) / total_population_size
    xms_fraction = np.sum(xms_populations) / total_population_size

    xss_fraction = 1.0 - xsm_fraction - xms_fraction - xmm_fraction

    if np.sum(population) == 0:
        xmm_fraction, xsm_fraction, xms_fraction, xss_fraction = 0, 0, 0, 0

    return np.array([xmm_fraction, xsm_fraction, xms_fraction, xss_fraction])

def compute_forces_of_infection(population, measles_states, rubella_states, t,
                                mbeta, rbeta, contact_structure, amplitude=0.3):
    # population_size_by_age = np.sum(population, axis=0)
    population_size_by_age = np.sum(population)

    force = 1.0 + amplitude * np.cos(2.0 * np.pi * t / 365.0)

    measles_sum = np.sum([population[s] for s in measles_states], axis=0)
    rubella_sum = np.sum([population[s] for s in rubella_states], axis=0)

    mfoi_top = np.array(np.dot(mbeta * contact_structure * force, measles_sum), dtype=float)
    rfoi_top = np.array(np.dot(rbeta * contact_structure * force, rubella_sum), dtype=float)

    foi_bottom = np.array(population_size_by_age, dtype=float)

    mfoi = np.divide(mfoi_top, foi_bottom, out=np.zeros_like(mfoi_top), where=foi_bottom != 0)
    rfoi = np.divide(rfoi_top, foi_bottom, out=np.zeros_like(rfoi_top), where=foi_bottom != 0)

    return mfoi, rfoi

def calculate_reported_cases(cases, reporting_ratio_true):
    ''' given an array of infection incidence and a reporting rate, it returns the time series of only the reported cases'''
    ni, nj, nk = cases.shape
    reported_cases = np.zeros((ni, nj, nk))

    for k in range(nk): # for each age
        for j in range(nj):  # for each case type
            for i in range(ni):  # for each day
                if cases[i, j, k] != 0:
                    # cases_today = cases[i, j, k]
                    rands = np.random.uniform( 0, 1, size=int(cases[i, j, k]) )
                    reported_cases[i, j, k] = np.sum(rands < reporting_ratio_true)
    return reported_cases

def estimate_coefficients(x, y):
    ''' fits slope with intercept = 0'''

    x = x.reshape((-1, 1))
    model = LinearRegression(fit_intercept=False).fit(x, y)
    slope = model.coef_
    intercept = model.intercept_

    return [intercept, slope]

def estimate_seasonality(x_data, y_data, maxfev=None):
    eps = 1e-6
    bounds = ([eps, eps, eps, 0, 0],
              [np.inf, np.inf, np.inf, np.inf, np.inf])

    starting_values = [1, 1, eps, 0.5, 0]
    ## Fit 5 params with log transform: a1, a2, beta, beta_amp, t0
    if np.all(y_data):
        starting_values, cov = curve_fit(f=five_parameter_log_transform_function,
                                         xdata=x_data,
                                         ydata=np.log(y_data),
                                         method='trf', bounds=bounds)

    # Fit 5 params: a1, a2, beta, beta_amp, t0
    params, covariance = curve_fit(f=five_parameter_function,
                                   xdata=x_data,
                                   ydata=y_data,
                                   bounds=bounds,
                                   p0=starting_values,
                                   maxfev=maxfev)
    return params

def five_parameter_log_transform_function(x_data, a1, a2, beta, beta_amp, t0):
    ''' I(t+1) = beta (1 + beta_a * cos(2 pi t / 26 + t0) * I^(a1) * S^(a2)
    t0 = 0 to 2 pi'''
    S, I, t = x_data
    return np.log(beta) + np.log(1 + beta_amp * np.cos(2*np.pi* t / 26 + t0)) \
           + a1 * np.log(I) + a2 * np.log(S)

def five_parameter_function(x_data, a1, a2, beta, beta_amp, t0):
    S, I, t = x_data
    return beta * (1 + beta_amp * np.cos(2 * np.pi * t / 26 + t0)) * I**a1 * S**a2

def load_demographics(NUMBER_OF_AGE_GROUPS=40):
    '''
    return age_dist: the population proportions by age (40 age groups) at equilibrium (after 300Y)
    return age_bin_sizes: the age bins of the 40 age groups
    return legend: the age group labels for the 40 age groups'''

    age_dist = np.array([0.00293521, 0.00302591, 0.00298253, 0.00295295, 0.00294309,
                         0.00293225, 0.00292436, 0.00292436, 0.00292337, 0.00293126,
                         0.00293521, 0.00293422, 0.00292929, 0.00291746, 0.00290168,
                         0.00288788, 0.00287408, 0.00286027, 0.00284844, 0.00283464,
                         0.00282281, 0.00281097, 0.00280506, 0.00279619, 0.09072319,
                         0.12810393, 0.10857599, 0.09216467, 0.07811078, 0.06625066,
                         0.05604399, 0.04736951, 0.04011481, 0.03397818, 0.0288571,
                         0.02439464, 0.02058292, 0.01749982, 0.01490773, 0.0826886])

    age_bin_sizes = np.ones(NUMBER_OF_AGE_GROUPS) * 5  ## default 5y age bins
    age_bin_sizes[0:24] = np.ones(24) * 1 / 12  # 24m monthly
    age_bin_sizes[24] = np.ones(1) * 3  # 2-5 years

    legends = ['0-1m', '1-2m', '2-3m', '3-4m', '4-5m',
               '5-6m', '6-7m', '7-8m', '8-9m', '9-10m',
               '10-11m', '11-12m', '12-13m', '13-14m', '14-15m',
               '15-16m', '16-17m', '17-18m', '18-19m', '19-20m',
               '20-21m', '21-22m', '22-23m', '23-24m',
               '2-5y',
               '5-10y', '10-15y', '15-20y', '20-25y', '25-30y',
               '30-35y', '35-40y', '40-45y', '45-50y', '50-55y',
               '55-60y', '60-65y', '65-70y', '70-75y', '75+']

    return age_dist, age_bin_sizes, legends


def load_country_demographics(country='Nigeria', NUMBER_OF_AGE_GROUPS =40):
    ''' loads the country-specific parameters
    age-specific fertility rates taken from: GBD 2017 Population and Fertility Collaborators, Lancet
    '''
    fertility_rates = np.zeros(NUMBER_OF_AGE_GROUPS)

    ## Nigeria
    fertility_rates[26:35] = np.array([2.3, 91.5, 202.4, 239.9, 219.2, 152.4, 82.5, 30.4, 0.58])  # 10-15Y, 15-20Y, ..., 50-55Y
    if country == 'Kenya':
        fertility_rates[26:35] = np.array([1.5, 70.9, 184.0, 153.7, 139.0, 85.0, 35.2, 6.4, 0.12])  # 10-15Y, 15-20Y, ..., 50-55Y

    fertility_rates_normalized = fertility_rates / np.sum(fertility_rates)

    return fertility_rates_normalized


def load_contact_matrix(contact_type='assorted', country='Nigeria'):
    ''' returns a matrix of 40x40 contact matrix (0-2Y monthly, 2-5, 5-10, 10-15, ..., 70-75, and 75+)
    expanded from a 16-age group matrix from Prem2017
    '''
    ## 16x16 contact matrix for Nigeria / Kenya
    CM = pd.read_csv('./data/contact_matrix_16_nigeria.csv', sep=',',header=None).to_numpy()
    if contact_type == 'assorted' and country == 'Kenya':
        CM = pd.read_csv('./data/contact_matrix_16_kenya.csv', sep=',',header=None).to_numpy()

    #### contact_matrix of 40 age groups (splits age class 0-4 to 25 age groups)
    CM2 = np.ones((40, 40))

    if contact_type == 'assorted':
        col1 = np.ones(25) * CM[0, 0]
        col1 = np.append(col1, CM[1:, 0])

        for i in range(25): ## first 25 columns
            CM2[:,i] = col1
        cval = 25
        for i in range(1, 16): ## columns 26 to 40
            temp = np.append(CM[0, i] * np.ones(25), CM[1:, i])
            CM2[:, cval] = temp
            cval += 1

    return CM2


def findBeta(R0, C, gamma, mu, totalPop):
    # compute the eignevalues of -F*V^(-1)
    [n1, n2] = np.shape(C)

    # create F (transmission matrix) and V (transition matrix)
    F = np.zeros((n1, n1))
    V = np.diag(-(gamma + mu) * np.ones(n1))
    N = np.sum(totalPop)

    for jvals in range(n1):
        F[jvals, :] = (C[jvals, :] / N) * totalPop[jvals]

    FV = np.dot(-F, np.linalg.inv(V))

    myEig = np.linalg.eig(FV)
    largestEig = np.max(myEig[0])

    ## double check that the calculated beta gives the right R0:
    # Fnew = beta * F
    # FVnew = np.dot(-Fnew, np.linalg.inv(V))
    # myEignew = np.linalg.eig(FVnew)
    # print('R0:', np.max(myEignew[0]))

    assert largestEig.imag == 0.0, 'largest eigenvalue is not real'
    beta = R0 / largestEig.real
    return beta


def compute_truncated_norm(bins = range(0, 24), interval = (0, 24), cutoff = 3, mu=9, sigma=1):
    a, b = interval
    if a > b:
        a, b = b, a

    # Compute the pdf
    distribution = truncnorm(-cutoff, cutoff, loc=mu, scale=sigma)
    pdf = [distribution.pdf(bin) for bin in bins]
    cdf = [distribution.cdf(bin) for bin in bins]

    return pdf, cdf


def simulate_msirv_equations_measles_only(initial_population, parameters):
    number_of_states, number_of_age_groups = np.shape(initial_population)
    cases = [np.zeros((4, number_of_age_groups))]

    age_bin_sizes, T, tau, tau_age = get_parameters_by_names(parameters, ["age_bin_sizes", "T", "tau", "tau_age"])
    populations = [initial_population]

    for current_time in T[1:]:
        population, case = msirv_equations_with_vacc_measles_only(populations[-1], parameters, current_time)
        if current_time % tau_age == 0:
            population = age_progression(population, [age_bin_sizes, tau_age])
        populations.append(population)
        cases.append(case)

    cases_by_type = np.reshape(cases, (len(T), 4, number_of_age_groups), order='F')

    return populations, cases_by_type


def endemic_equilibrium_for_measles_only_after_300Y_uniform(total_population_size, number_of_classes=32, number_of_age_groups=40):
    ''' Measles only endemic equilibrium after 300 years with: UNIFORM contact structure
        N0 = 1 million
        mR0, rR0 = 15, 0
        mdelta, rdelta = 200 * mR0 / N0, 0
        births = 27 / 1000 / 365 (normalized by Kenya-like age-specific fertility)
        '''
    # initial_population = populations[-1]
    # total_population = np.sum(initial_population)
    # age_distribution = np.sum(initial_population, 0) / total_population
    # class_distribution = np.sum(initial_population, 1) / total_population

    age_distribution = np.array([0.00112872, 0.00221418, 0.0022014 , 0.00220927, 0.00221812,
       0.0022191 , 0.00222008, 0.00221812, 0.00221713, 0.00222008,
       0.00222697, 0.002225  , 0.00222402, 0.0022191 , 0.0022073 ,
       0.0021955 , 0.00218075, 0.00217092, 0.00216306, 0.00215519,
       0.00214929, 0.00213651, 0.00213258, 0.00213258, 0.07133074,
       0.10451401, 0.09215213, 0.08114512, 0.07140547, 0.06276897,
       0.0553546 , 0.04882708, 0.04310875, 0.03799508, 0.03339465,
       0.02946379, 0.02595472, 0.02282517, 0.02006432, 0.14811042])

    class_distribution = np.array([0.        , 0.        , 0.        , 0.        , 0.00535357,
       0.0695767 , 0.00325835, 0.92181138, 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        ])

    temp = np.ones((number_of_classes, number_of_age_groups)) * age_distribution
    normalized_population_at_EE = np.transpose(np.transpose(temp) * class_distribution)
    population_at_endemic_equilbrium = np.round(normalized_population_at_EE * total_population_size, 0)

    ## Add rounding errors to fully recovered folks age 30 to 35YOs
    roundingError = total_population_size - np.sum(np.round(population_at_endemic_equilbrium, 0))
    index_age = 30  ## 30-35Y age group
    index_state = 7  ## XRS
    population_at_endemic_equilbrium[index_state, index_age] += roundingError
    return population_at_endemic_equilbrium


def endemic_equilibrium_for_measles_only_after_300Y(total_population_size, number_of_classes=32, number_of_age_groups=40):
    ''' Measles only endemic equilibrium after 300 years with: Kenya-like contact structure, N0 = 1 million, mR0 = 15,
        mdelta = 200 * mR0 / N0, annual births = 27 per 1000 (Kenya-like age-specific fertility), amplitude = 0.3
    '''

    age_distribution = np.array([0.00114832, 0.00223505, 0.00220624, 0.00221121, 0.00220823,
                                0.00220426, 0.00220624, 0.00220525, 0.00220724, 0.00220525,
                                0.00220227, 0.00219234, 0.0021824 , 0.00217048, 0.00216154,
                                0.00215558, 0.00215061, 0.00215459, 0.00215757, 0.00216154,
                                0.00216452, 0.00216353, 0.00216253, 0.00215757, 0.07123054,
                                0.10433014, 0.0919569 , 0.08099522, 0.07131696, 0.06283271,
                                0.05531798, 0.04882739, 0.04293382, 0.03781903, 0.03337178,
                                0.0293765 , 0.02592161, 0.02280545, 0.02013134, 0.14945827])

    class_distribution = np.array([0., 0., 0., 0.,
                                5.44160113e-03, 3.58750319e-02, 1.27149497e-04, 9.58556217e-01,
                                0., 0., 0., 0.,
                                0., 0., 0., 0.,
                                0., 0., 0., 0.,
                                0., 0., 0., 0.,
                                0., 0., 0., 0.,
                                0., 0., 0., 0.])

    temp = np.ones((number_of_classes, number_of_age_groups)) * age_distribution
    normalized_population_at_EE = np.transpose(np.transpose(temp) * class_distribution)
    population_at_endemic_equilbrium = np.round(normalized_population_at_EE * total_population_size, 0)

    ## Add rounding errors to fully recovered folks age 30 to 35YOs
    roundingError = total_population_size - np.sum(np.round(population_at_endemic_equilbrium, 0))
    index_age = 30  ## 30-35Y age group
    index_state = 7  ## XRS
    population_at_endemic_equilbrium[index_state, index_age] += roundingError
    return population_at_endemic_equilbrium


## Estimate the reporting rate from a set of time series and create a plot
def estimate_reporting_rates_and_seasonality():
    ''' This estimates reporting rates, amplitude and phase shifts '''
    subtract_young_cases = True
    erase_zeros = True
    fit_amplitude_and_phase_shift = False

    days_counted = (365 * 10 // 14) * 14  ## days in 10 years dividable into infectious generations
    interval_counted = (26 * 10) * 14  ## 10 year of infectious periods
    age_in_months = np.arange(1, 41, 1)
    age_in_months[24:] = np.arange(5, 80 + 1, 5) * 12 # 5-year age bins after 24 months

    phase_shifts = [0, 180]
    amps = [0.3, 0.4]
    vaccination_coverages = [0.0, 0.5, 0.7, 0.9]

    reporting_ratios_true = [0.01, 0.05, 0.10, 0.20]
    vlength = len(vaccination_coverages)
    rlength = len(reporting_ratios_true)
    number_of_intervals = 10
    colors = sns.color_palette("colorblind")
    shapes = ['v', '^', 'o', 's']

    plt.figure(1, figsize=(12, 8), layout='tight')
    plt.suptitle('Estimated reporting rates (case corrections)')
    tic = time.time()
    num=0
    for p, phase_shift in enumerate(phase_shifts):
        for a, amp in enumerate(amps):
            # reset the parameters of interest
            estimated_reporting_rates = np.zeros((len(reporting_ratios_true), len(vaccination_coverages) + 1, number_of_intervals))
            params = np.zeros((len(reporting_ratios_true), len(vaccination_coverages), 5))
            estimated_amplitudes = np.zeros((len(reporting_ratios_true), len(vaccination_coverages) + 1, number_of_intervals))
            estimated_phase_shifts = np.zeros((len(reporting_ratios_true), len(vaccination_coverages) + 1, number_of_intervals))
            Ns = np.zeros((len(vaccination_coverages), number_of_intervals))
            #
            for i, vc in enumerate(vaccination_coverages):
                print(f"phase shift {phase_shift}, amplitude {amp}, VC: {vc}, time = {int(time.time() - tic)}")
                # read the file
                filename = 'sim_Kenya_amp' + str(int(amp * 100)) + '_VC=' + str(vc) + \
                           '_uniform_theta_measlesOnly_400Y_normalvacc_sigma=0.5.pkl'
                #
                populations_main, cases_all_main, parameters, _ = read_from_pickle(filename)
                #
                start_time = 0 + phase_shift
                for d in range(number_of_intervals):
                    end_time = start_time + days_counted
                    print(f"  Loop {d}: (Start, End) = ({start_time}, {end_time})")

                    ## Create the infections data for this interval. Calculate the cumulative births
                    cases_all = cases_all_main[start_time:end_time, :, :]
                    Ns[i, d] = int(np.sum(populations_main, (2, 1))[start_time:end_time].mean())
                    #
                    births_per_day = (Ns[i, d] * parameters["birth_rate"]) * (1 - vc * parameters["vaccine_eff_m"])
                    cum_births = (births_per_day * np.ones(interval_counted)).reshape( (-1, 14) )

                    ## create the reported cases
                    for j, rr in enumerate(reporting_ratios_true):
                        cases_reported = calculate_reported_cases(cases_all[-days_counted:], rr)
                        cases_reported_all = np.sum(np.sum(cases_reported, -1), -1)
                        cases_reported_under_9_times_vc = np.zeros(days_counted)
                        #
                        if subtract_young_cases == True:
                            cases_reported_under_9 = np.sum(cases_reported[:, (0, 2), :9], (1, 2))
                            for day, ncases in enumerate(cases_reported_under_9):
                                if ncases > 0.0:
                                    rands = np.random.uniform(0, 1, size=(int(ncases)))
                                    cases_reported_under_9_times_vc[day] = np.sum(rands < vc * parameters['vaccine_eff_m'])
                        #
                        all_FR_cases = cases_reported_all - cases_reported_under_9_times_vc
                        FR_cases = np.sum(all_FR_cases.reshape((-1, 14)), -1)
                        temp = estimate_coefficients(x=np.cumsum(np.sum(cum_births, -1)),
                                                     y=np.cumsum(FR_cases))
                        estimated_reporting_rates[j, 0, d] = rr
                        estimated_reporting_rates[j, i + 1, d] = temp[1]
                        #
                        if fit_amplitude_and_phase_shift == True:
                            ## estimate seasonality within rr loop
                            I_data = FR_cases[:-1] / temp[1]
                            Y_data = FR_cases[1:] / temp[1]
                            B_data = np.sum(cum_births, 1)[1:]
                            T_data = np.arange(len(I_data))

                            residuals = np.cumsum(B_data) - np.cumsum(I_data)
                            S_mean = 70000  ## found by GLM
                            S_data = S_mean + residuals
                            if erase_zeros == True:
                                nonzeroes = (S_data != 0) & (I_data != 0) & (Y_data != 0)
                                S_data = S_data[nonzeroes]
                                I_data = I_data[nonzeroes]
                                Y_data = Y_data[nonzeroes]
                                T_data = T_data[nonzeroes]
                            params[j, i, :] = estimate_seasonality(x_data=np.array([S_data, I_data, T_data]),
                                                                   y_data=Y_data)
                            estimated_amplitudes[j, 0, d] = amp
                            estimated_amplitudes[j, i, d] = params[j, i, 3]
                            estimated_phase_shifts[j, 0, d] = phase_shift
                            estimated_phase_shifts[j, i, d] = params[j, i, 4]
                    # reset start_time for next loop
                    start_time += 3650
            # plot the estimated reporting rates for this amplitude
            data = estimated_reporting_rates[:, 1:, :]
            data_mean = np.zeros((rlength, vlength))
            data_min = np.zeros((rlength, vlength))
            data_max = np.zeros((rlength, vlength))
            for r in range(rlength):
                for v in range(vlength):
                    data_mean[r, v] = data[r, v, :].mean()
                    data_min[r, v] = data[r, v, :].min()
                    data_max[r, v] = data[r, v, :].max()
            #
            plt.figure(1)  ## estimated reporting rates
            for ii, vc in enumerate(vaccination_coverages):
                for jj, rr in enumerate(reporting_ratios_true):
                    plt.subplot(rlength, vlength, (vlength * jj) + (ii + 1))
                    plt.title(f'{int(vc * 100)}% VC, r = {np.round(rr, 1)}', fontsize=10)
                    data_mean = data[jj, ii, :].mean()
                    xerror = [[data_mean - data[jj, ii, :].min()],
                              [data[jj, ii, :].max() - data_mean]]
                    plt.errorbar(data_mean, num,
                                 xerr=xerror,
                                 fmt=shapes[p],
                                 markersize=3,
                                 color=colors[a],
                                 elinewidth=1)
                    if a == 0: plt.axvline(x=rr, color='k', lw=0.5, linestyle='--')
                    plt.xlim([0, 0.3])
                    plt.yticks([])
                    plt.xticks(fontsize=10)
                    plt.yticks(fontsize=10)
                    if a == 0 and ii == 0:
                        plt.ylabel('Estimated', fontsize=10)
                    if a == 0 and jj == 3:
                        plt.xlabel('True', fontsize=10)
            num+=1  ## counter for the plotting

    return []


if __name__ == "__main__":
    ## Create the time series of cases
    NUMBER_OF_AGE_GROUPS = 24 + 16
    N0 = 1_000_000
    age_dist, age_bin_sizes, legend = load_demographics()  ## endemic age distribution (40-groups) + age group legends
    N_by_age_groups = age_dist * N0

    ## demography parameters
    fertility_rates_normalized = load_country_demographics("Kenya")
    birth_rate = (40 / 1000) / 365
    death_rates = np.ones(NUMBER_OF_AGE_GROUPS) * birth_rate

    C = load_contact_matrix('uniform')

    reporting_ratios = 1 * np.ones(NUMBER_OF_AGE_GROUPS)

    ## measles parameters
    mR0 = 15
    mgamma = 1 / 14  # recovery rate
    mbeta = findBeta(mR0, C, mgamma, death_rates[0], N_by_age_groups)
    msigma = np.ones(NUMBER_OF_AGE_GROUPS)  # rate of loss of maternal immunity
    msigma[0:4] = 1 / 120
    mdelta = 200 * mR0 / N0

    ## rubella parameters (rubella is turned off)
    rR0 = 0
    rgamma = 0
    rbeta = findBeta(rR0, C, rgamma, death_rates[0], N_by_age_groups)
    rsigma = np.zeros(NUMBER_OF_AGE_GROUPS)
    rdelta = 0

    ## routine vaccination parameters
    vc_goal = 0.
    VEM = 1
    VER = 0

    average_month_of_vacc = 9  # months
    months_eligible_for_vacc = np.arange(0, 24)  # age in months

    pdf, cdf = compute_truncated_norm(mu=average_month_of_vacc, sigma=0.5)

    theta = - np.log(1 - vc_goal) / cdf[-1]
    vc_goal_probabilities = list(pdf) + [0 for _ in range(24, 40)]
    vc_time = 30
    vc_goal_rates = np.multiply(theta / vc_time, vc_goal_probabilities)

    ## initial conditions: age-distributed XSS with only measles
    INPUT = np.zeros((32, NUMBER_OF_AGE_GROUPS))
    INPUT[XSS, :] = np.floor(N0 * age_dist * 1 / mR0)
    INPUT[XIS, 24] = 5  # XIS, 2-5 years old
    INPUT[XRS, :] = np.floor(N0 * age_dist * (1 - 1 / mR0))

    # INPUT = endemic_equilibrium_for_measles_only_after_300Y(total_population_size=N0)  ## without vaccination
    INPUT = endemic_equilibrium_for_measles_only_after_300Y_uniform(
        total_population_size=N0)  ## without vaccination

    ## time steps
    tau = 1.0
    tau_age = 15.0

    ## transient run
    t_start = 0.0
    t_end = 365 * 10
    t_transient = np.arange(t_start, t_end + tau, tau)

    parameters = {
        "age_bin_sizes": age_bin_sizes,
        "death_rates": death_rates,
        "birth_rate": birth_rate,
        "contact_matrix": C,
        "fertility_rates_normalized": fertility_rates_normalized,
        "importation_rate_m": mdelta,
        "importation_rate_r": rdelta,
        "maternal_immunity_waning_rate_m": msigma,
        "maternal_immunity_waning_rate_r": rsigma,
        "recovery_rate_m": mgamma,
        "recovery_rate_r": rgamma,
        "reporting_ratios": reporting_ratios,
        "transmission_rate_m": mbeta,
        "transmission_rate_r": rbeta,
        "vaccine_coverage": vc_goal_rates,
        "vaccine_eff_m": VEM,
        "vaccine_eff_r": VER,
        "T": t_transient,
        "tau": tau,
        "tau_age": tau_age
    }

    vaccination_rates = [0, 0.2, 0.4, 0.6, 0.8, 0.9]
    for i, vc in enumerate(vaccination_rates):
        print(i, vc)
        ## update new theta and recalculate vaccination probabilities
        theta = - np.log(1 - vc) / sum(pdf)
        vc_goal_rates = np.multiply(theta / vc_time, vc_goal_probabilities)
        parameters["vaccine_coverage"] = vc_goal_rates

        populations, cases = simulate_msirv_equations_measles_only(INPUT, parameters)

        ## save results
        mylist = [populations, cases]

