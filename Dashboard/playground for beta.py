from scipy.optimize import minimize, minimize_scalar
#1) OBJECTIVE FUNCTION: define some equation
def equation(x):
    vasicek_fit = np.array(Vasicek_fit(x, std_TTCPd_pred_std)).ravel()
    original_TTC_PDs = df["TTC PDs"].values.flatten()
    return sum(vasicek_fit - original_TTC_PDs) ** 2

minimize_scalar(equation(x = beta))
#2) DEFINE CONSTRAINTS
x0 = [0,0.5,1]
result = minimize(equation, x0, method='Nelder-Mead', options={'xtol': 1e-8, 'disp': True})
print(result.x)

from scipy.optimize import differential_evolution
from scipy.optimize import curve_fit
# objective function
def func(x, a, b):
    return a * x + b

def sumOfSquaredError(parameterTuple):
    val = func(original_TTC_PDs, *parameterTuple)
    return np.sum((vasicek_fit - val) ** 2.0)


def generate_Initial_Parameters():
    # min and max used for bounds
    maxX = max(original_TTC_PDs)
    minX = min(original_TTC_PDs)
    maxY = max(vasicek_fit)
    minY = min(vasicek_fit)

    diffY = maxY - minY
    diffX = maxX - minX

    parameterBounds = []
    parameterBounds.append([0.0, diffY]) # search bounds for amplitude
    parameterBounds.append([minX, maxX]) # search bounds for center

    # "seed" the numpy random number generator for repeatable results
    result = differential_evolution(sumOfSquaredError, parameterBounds, seed=3)
    return result.x

# by default, differential_evolution completes by calling curve_fit() using parameter bounds
geneticParameters = generate_Initial_Parameters()

fittedParameters, pcov = curve_fit(func, original_TTC_PDs, vasicek_fit, geneticParameters)
print('Fitted parameters:', fittedParameters)
modelPredictions = func(original_TTC_PDs, *fittedParameters) 