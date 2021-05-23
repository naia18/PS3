# ------------ MAIN LIBRARIES TO USE ------------ 

using CSV, DataFrames
using Plots
using Distributions, Statistics
using Optim, NLopt
using LinearAlgebra
using Dates
import Random

Random.seed!(14052021)

# ------- DATA COLLECTION FROM DATASET: Population of 2 countries -----------

df = DataFrame(CSV.File("worldcities.csv"))
hiriak = df.country                          # Read the countries in dataset

# Data from Country 1
bai1 = (hiriak.=="Japan")                    # Choose Country1 to analyse
population1 = df.population[bai1]
cities1 = df.city[bai1]

# Data from Country 2
bai2 = (hiriak.=="Russia")                    # Choose Country2 to analyse
population2 = df.population[bai2]
cities2 = df.city[bai2]

df1 = DataFrame(Cities =cities1,Population=population1)  # Construct the dataframe out of the data
data1 = convert(Matrix, df1);
datuak1 = data1[5:170,2];
df2 = DataFrame(Cities =cities2,Population=population2)  # Construct the dataframe out of the data
data2 = convert(Matrix, df2);
datuak2 = data2[5:170,2];

histogram([datuak1,datuak2], bins=20, alpha=0.8,label=["Japan" "Russia"], title="Histogram population by cities")

# ---------- MAIN FUNCTIONS --------------

function Gibbs(reps::Int, burnin::Int,data)
    chain = fill(NaN, (reps,2))
    alpha   = 1.
    c   = 100.
    n_data = size(data,1)
    x_star = minimum(data)
    for i = 1:reps
        for j = 1:burnin
            alpha = rand(Gamma(n_data+1,1.0./(sum(log.(data))-n_data*log(c))))
            c = x_star*rand(Uniform(0,1))^(1/(n_data*alpha+1))
        end
        chain[i,:] = [alpha,c]     # Construct
    end
    return chain
end
    
# Monomial Distribution
function Monomial(a,x_star,n_data)
    ran = range(0,x_star,length=1000)
    c = zeros(size(ran,1))
    c = ran.^(a*n_data)
end    
        

# Compute Gibbs Sampling and Plot 
chain1 = Gibbs(10000,100,datuak1);
chain2 = Gibbs(10000,100,datuak2);
plot([chain1, chain2],layout=2,alpha=0.8,seriestype=[:histogram],label=["Japan" "Japan" "Russia" "Russia"],title=["α" "c"],size=(1000,500))

# Country 1
n1 = size(chain1,1)
m1 = zeros(n1,2)
for i=1:n1
    m1[i,:] = mean(chain1[1:i,:],dims=1);
end

# Country 2
n2 = size(chain2,1)
m2 = zeros(n2,2)
for i=1:n2
    m2[i,:] = mean(chain2[1:i,:],dims=1);
end

plot([m1,m2],layout=2,lw=2,title=["<α>" "<c>"],label=["Japan" "Japan" "Russia" "Russia"],size=(1000,500))

reps=200;

# ---- Country 1 ------
alpha1  = m1[end,1];
c1 = m1[end,2];
chain_trial1 = fill(NaN, (reps,1))
max_val1 = maximum(datuak1)
for i=1:reps
    chain_trial1[i] = rand(Pareto(alpha1,c1))    # Create Pareto sampling distribution from converged \alpha, c
end

# ----- Country 2 ------
alpha2  = m2[end,1];
c2 = m2[end,2];
chain_trial2 = fill(NaN, (reps,1))
max_val2 = maximum(datuak2)
for i=1:reps
    chain_trial2[i] = rand(Pareto(alpha2,c2))   # Create Pareto sampling distribution from converged \alpha, c
end

# PLOT
histogram([chain_trial1[chain_trial1.<maximum(datuak1)],chain_trial2[chain_trial2.<maximum(datuak2)]],bins=20,alpha=0.8,title="Data generation",label=["Japan Simulation" "Russia Simulation"])   # We focus our attention on the data

histogram([datuak1,chain_trial1[chain_trial1.<maximum(datuak1)]],layout=1,bins=30,alpha=0.7,color= [:cyan :orange],size=(900,500),title="Japan population distribution",label=["Real data" "Generated data"] )   # We focus our attention on the data

histogram([datuak2,chain_trial2[chain_trial2.<maximum(datuak2)]],layout=1,bins=30,alpha=0.7,color= [:cyan :orange],size=(900,500),title="Russia population distribution",label=["Real data" "Generated data"] )   # We focus our attention on the data

# ------------- MAIN FUNCTIONS TO USE --------------------

function fmincon(obj, startval, R=[], r=[], lb=[], ub=[]; tol = 1e-10, iterlim=0)
    # the objective is an anonymous function
    function objective_function(x::Vector{Float64}, grad::Vector{Float64})
        obj_func_value = obj(x)[1,1]
        return(obj_func_value)
    end
    # impose the linear restrictions
    function constraint_function(x::Vector, grad::Vector, R, r)
        result = R*x .- r
        return result[1,1]
    end
    opt = Opt(:LN_COBYLA, size(startval,1))
    min_objective!(opt, objective_function)
    # impose lower and/or upper bounds
    if lb != [] lower_bounds!(opt, lb) end
    if ub != [] upper_bounds!(opt, ub) end
    # impose linear restrictions, by looping over the rows
    if R != []
        for i = 1:size(R,1)
            equality_constraint!(opt, (theta, g) -> constraint_function(theta, g, R[i:i,:], r[i]), tol)
        end
    end    
    xtol_rel!(opt, tol)
    ftol_rel!(opt, tol)
    maxeval!(opt, iterlim)
    (objvalue, xopt, flag) = NLopt.optimize(opt, startval)
    return xopt, objvalue, flag
end

# --------------- DATASET ---------------

data = DataFrame(CSV.File("GOOG.csv",header=true))
first(data,6)
y = data[!,"Adj Close"];               # Take closing price
y = 100.0*diff(log.(y));               # Compute log-returns
n = size(y,1);
last(data,7)

function garch11(theta, y)
    # dissect the parameter vector
    mu = theta[1]
    rho = theta[2]
    omega = theta[3]
    alpha = theta[4]
    beta = theta[5]
    resid = y[2:end] .- mu .- rho*y[1:end-1]
    n = size(resid,1)
    h = zeros(n)
    # initialize variance; either of these next two are reasonable choices
    h[1] = var(y)
    #h[1] = var(y)
    rsq = resid.^2.0
    for t = 2:n
        h[t] = omega + alpha*rsq[t-1] + beta*h[t-1]
    end
    logL = -log(sqrt(2.0*pi)) .- 0.5*log.(h) .- 0.5*rsq./h    # h = \sigma^2
end

function garch12(theta, y)
    # dissect the parameter vector
    mu = theta[1]
    rho = theta[2]
    omega = theta[3]
    alpha1 = theta[4]
    alpha2 = theta[5]
    beta1 = theta[6]
    resid = y[2:end] .- mu .- rho*y[1:end-1]
    n = size(resid,1)
    h = zeros(n)
    # initialize variance; either of these next two are reasonable choices
    h[1] = var(y)
    h[2] = var(y)
    #h[2] = omega + alpha1*rsq[1] + beta1*h[1]
    #h[1] = var(y)
    rsq = resid.^2.0
    for t = 3:n
        h[t] = omega + alpha1*rsq[t-1] + alpha2*rsq[t-2] + beta1*h[t-1]
    end
    logL = -log(sqrt(2.0*pi)) .- 0.5*log.(h) .- 0.5*rsq./h    # h = \sigma^2
end

function garch21(theta, y)
    # dissect the parameter vector
    mu = theta[1]
    rho = theta[2]
    omega = theta[3]
    alpha1 = theta[4]
    beta1 = theta[5]
    beta2 = theta[6]
    resid = y[2:end] .- mu .- rho*y[1:end-1]
    n = size(resid,1)
    h = zeros(n)
    # initialize variance; either of these next two are reasonable choices
    h[1] = var(y)
    #h[2] = omega + alpha1*rsq[1] + beta1*h[1]
    h[2] = var(y)
    rsq = resid.^2.0
    for t = 3:n
        h[t] = omega + alpha1*rsq[t-1] + beta1*h[t-1] + beta2*h[t-2]
    end
    logL = -log(sqrt(2.0*pi)) .- 0.5*log.(h) .- 0.5*rsq./h    # h = \sigma^2
end

function garch22(theta, y)
    # dissect the parameter vector
    mu = theta[1]
    rho = theta[2]
    omega = theta[3]
    alpha1 = theta[4]
    alpha2 = theta[5]
    beta1 = theta[6]
    beta2 = theta[7]
    resid = y[2:end] .- mu .- rho*y[1:end-1]
    n = size(resid,1)
    h = zeros(n)
    # initialize variance; either of these next two are reasonable choices
    h[1] = var(y)
    #h[2] = omega + alpha1*rsq[1] + beta1*h[1]
    h[2] = var(y)
    rsq = resid.^2.0
    for t = 3:n
        h[t] = omega + alpha1*rsq[t-1] + alpha2*rsq[t-2] + beta1*h[t-1] + beta2*h[t-2]
    end
    logL = -log(sqrt(2.0*pi)) .- 0.5*log.(h) .- 0.5*rsq./h    # h = \sigma^2
end

# ----- GARCH (1,1) --------
thetastart11 = [mean(y); 0.0; var(y); 0.1; 0.1]
obj = theta -> -sum(garch11(theta, y))
thetahat11, logL11, junk  = fmincon(obj, thetastart11, [], [], [-Inf, -1.0, 0.0, 0.0, 0.0], [Inf, 1.0, Inf, 1.0, 1.0])
thetahat11 = [thetahat11[1:4]; NaN; thetahat11[5]; NaN]
# ----- GARCH (1,2) --------
thetastart12 = [mean(y); 0.0; var(y); 0.1; 0.1; 0.1]
obj = theta -> -sum(garch12(theta, y))
thetahat12, logL12, junk  = fmincon(obj, thetastart12, [], [], [-Inf, -1.0, 0.0, 0.0, 0.0, 0.0], [Inf, 1.0, Inf, 1.0, 1.0, 1.0])
thetahat12 = [thetahat12; NaN]
# ----- GARCH (2,1) --------
thetastart21 = [mean(y); 0.0; var(y); 0.1; 0.1; 0.1]
obj = theta -> -sum(garch21(theta, y))
thetahat21, logL21, junk  = fmincon(obj, thetastart21, [], [], [-Inf, -1.0, 0.0, 0.0, 0.0, 0.0], [Inf, 1.0, Inf, 1.0, 1.0, 1.0])
thetahat21 = [thetahat21[1:4];NaN;thetahat21[5:end]]
# ----- GARCH (2,2) --------
thetastart22 = [mean(y); 0.0; var(y); 0.1; 0.1; 0.1; 0.1]
obj = theta -> -sum(garch22(theta, y))
thetahat22, logL22, junk  = fmincon(obj, thetastart22, [], [], [-Inf, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [Inf, 1.0, Inf, 1.0, 1.0, 1.0, 1.0])

df = DataFrame(Garch11=thetahat11, Garch12=thetahat12, Garch21=thetahat21, Garch22=thetahat22)

AIC = zeros(4)
AIC[1] = -2*logL11 + 2*5;
AIC[2] = -2*logL12 + 2*6;
AIC[3] = -2*logL21 + 2*6;
AIC[4] = -2*logL22 + 2*7;
df = DataFrame(Models = ["Garch(1,1)", "Garch(1,2)", "Garch(2,1)", "Garch(2,2)"], AIC_Value=AIC)

BIC = zeros(4)
BIC[1] = -2*logL11 + log(n)*5;
BIC[2] = -2*logL12 + log(n)*6;
BIC[3] = -2*logL21 + log(n)*6;
BIC[4] = -2*logL22 + log(n)*7;
df = DataFrame(Models = ["Garch(1,1)", "Garch(1,2)", "Garch(2,1)", "Garch(2,2)"], BIC_Value=BIC)


