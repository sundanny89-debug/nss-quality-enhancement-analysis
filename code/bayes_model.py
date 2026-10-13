"""Cohort-aware Bayesian hierarchical model for NSS benchmark gaps.

Observation model (pre-analysis plan):  D_pqt ~ Normal(theta_pqt, s_pqt^2)
Latent model (M1, primary):
    theta_pqt = mu + alpha_q + gamma_t + u_p + v_pq + c_pt + eps_pqt
      alpha_q  : theme effect                       (7)
      gamma_t  : year effect                        (4)
      u_p      : persistent unit (provider/subject) effect
      v_pq     : persistent unit x theme effect
      c_pt     : unit x year cohort shock, shared across themes   <-- mandatory (rule 10)
      eps_pqt  : idiosyncratic theme-year deviation (overdispersion beyond OfS sampling SD)
    eps is marginalised analytically: D ~ Normal(m, s^2 + sigma_eps^2), and recovered post hoc.
Variants: M0 drops c_pt (for comparison); M2 adds a unit-specific linear trend b_p*(t - tbar).

Priors (proposed for freeze; all on the percentage-point scale):
    mu ~ N(0, 5);  sigma_alpha, sigma_u, sigma_v, sigma_c, sigma_eps ~ HalfNormal(5);  sigma_gamma ~ HalfNormal(2);
    sigma_b ~ HalfNormal(2) (M2 only).  Non-centred parameterisation throughout.
"""
import numpy as np, pandas as pd, jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, Predictive

numpyro.set_host_device_count(1)

PRIORS = dict(mu_sd=5.0, sd_alpha=5.0, sd_gamma=2.0, sd_u=5.0, sd_v=5.0, sd_c=5.0, sd_eps=5.0, sd_b=2.0)


def build_index(df, unit_col, theme_col='theme', year_col='Year'):
    """Create integer indices for the model. Returns dict of arrays and lookup tables."""
    d = df.copy().reset_index(drop=True)
    units = pd.Index(sorted(d[unit_col].unique()))
    themes = pd.Index(sorted(d[theme_col].unique()))
    years = pd.Index(sorted(d[year_col].unique()))
    d['u_i'] = units.get_indexer(d[unit_col])
    d['q_i'] = themes.get_indexer(d[theme_col])
    d['t_i'] = years.get_indexer(d[year_col])
    uq = d[['u_i', 'q_i']].drop_duplicates().sort_values(['u_i', 'q_i']).reset_index(drop=True)
    uq['v_i'] = np.arange(len(uq))
    ut = d[['u_i', 't_i']].drop_duplicates().sort_values(['u_i', 't_i']).reset_index(drop=True)
    ut['c_i'] = np.arange(len(ut))
    d = d.merge(uq, on=['u_i', 'q_i']).merge(ut, on=['u_i', 't_i'])
    d = d.sort_values(['u_i', 'q_i', 't_i']).reset_index(drop=True)
    return dict(df=d, units=units, themes=themes, years=years, uq=uq, ut=ut,
                n_u=len(units), n_q=len(themes), n_t=len(years), n_v=len(uq), n_c=len(ut))


def _sd_prior(name, p, informative):
    """HalfNormal(scale) by default; if `informative` gives (median, sd_log) for this name, use LogNormal."""
    if informative and name in informative:
        med, sdl = informative[name]
        return numpyro.sample(name, dist.LogNormal(jnp.log(med), sdl))
    return numpyro.sample(name, dist.HalfNormal(p[name]))


def model(u_i, q_i, t_i, v_i, c_i, s, tcen, n_u, n_q, n_t, n_v, n_c, D=None,
          cohort=True, trend=False, priors=PRIORS, informative=None, dynamic=False,
          eps=True, alpha_prior='hier', x=None, rho_err=False):
    """eps=False removes the idiosyncratic theme-year term (guide specification);
       alpha_prior='fixed' uses alpha_q ~ N(0,3) with a sum-to-zero constraint (guide) instead of a hierarchical scale;
       x: optional standardised covariate (e.g. response rate) with slope beta ~ N(0,2)."""
    p = priors
    mu = numpyro.sample('mu', dist.Normal(0., p['mu_sd']))
    sd_gamma = _sd_prior('sd_gamma', p, informative)
    sd_u = _sd_prior('sd_u', p, informative)
    sd_v = _sd_prior('sd_v', p, informative)
    if eps:
        sd_eps = _sd_prior('sd_eps', p, informative)
    else:
        sd_eps = numpyro.deterministic('sd_eps', jnp.array(0.0))
    if alpha_prior == 'fixed':
        with numpyro.plate('theme_free', n_q - 1):
            alpha_free = numpyro.sample('alpha_free', dist.Normal(0., p.get('alpha_fixed_sd', 3.0)))
        alpha_z = None
        alpha = numpyro.deterministic('alpha', jnp.concatenate([alpha_free, -jnp.sum(alpha_free, keepdims=True)]))
    else:
        sd_alpha = _sd_prior('sd_alpha', p, informative)
        with numpyro.plate('theme', n_q):
            alpha_z = numpyro.sample('alpha_z', dist.Normal(0., 1.))
        alpha = numpyro.deterministic('alpha', alpha_z * sd_alpha)
    with numpyro.plate('year', n_t):
        gamma_z = numpyro.sample('gamma_z', dist.Normal(0., 1.))
    with numpyro.plate('unit', n_u):
        u_z = numpyro.sample('u_z', dist.Normal(0., 1.))
    with numpyro.plate('unit_theme', n_v):
        v_z = numpyro.sample('v_z', dist.Normal(0., 1.))
    gamma = numpyro.deterministic('gamma', gamma_z * sd_gamma)
    u = numpyro.deterministic('u', u_z * sd_u)
    v = numpyro.deterministic('v', v_z * sd_v)
    m = mu + alpha[q_i] + gamma[t_i] + u[u_i] + v[v_i]
    if dynamic:
        # M3: unit x theme effect follows a random walk across years: w_{pq,t} = v_pq + sum_{k<=t} eta_{pq,k}
        sd_eta = numpyro.sample('sd_eta', dist.HalfNormal(p.get('sd_eta', 2.0)))
        with numpyro.plate('unit_theme_rw', n_v):
            with numpyro.plate('year_rw', n_t - 1):
                eta_z = numpyro.sample('eta_z', dist.Normal(0., 1.))     # (n_t-1, n_v)
        eta = jnp.concatenate([jnp.zeros((1, n_v)), jnp.cumsum(eta_z * sd_eta, axis=0)], axis=0)  # (n_t, n_v)
        w_dyn = numpyro.deterministic('w_dyn', eta)
        m = m + w_dyn[t_i, v_i]
    if cohort:
        sd_c = _sd_prior('sd_c', p, informative)
        with numpyro.plate('unit_year', n_c):
            c_z = numpyro.sample('c_z', dist.Normal(0., 1.))
        c = numpyro.deterministic('c', c_z * sd_c)
        m = m + c[c_i]
    if x is not None:
        beta = numpyro.sample('beta', dist.Normal(0., 2.0))
        m = m + beta * x
    if trend:
        sd_b = numpyro.sample('sd_b', dist.HalfNormal(p['sd_b']))
        with numpyro.plate('unit_b', n_u):
            b_z = numpyro.sample('b_z', dist.Normal(0., 1.))
        b = numpyro.deterministic('b', b_z * sd_b)
        m = m + b[u_i] * tcen
    numpyro.deterministic('m', m)
    if rho_err:
        # M4a: guide latent structure + equicorrelated same-provider-year observation error (rho), as in M4
        rho = numpyro.sample('rho', dist.Beta(2., 2.))
        with numpyro.plate('unit_year_g', n_c):
            g = numpyro.sample('g', dist.Normal(0., 1.))
        with numpyro.plate('obs', len(s)):
            numpyro.sample('D', dist.Normal(m + s * jnp.sqrt(rho) * g[c_i], jnp.sqrt(s ** 2 * (1. - rho) + sd_eps ** 2)), obs=D)
    else:
        with numpyro.plate('obs', len(s)):
            numpyro.sample('D', dist.Normal(m, jnp.sqrt(s ** 2 + sd_eps ** 2)), obs=D)


def fit(idx, cohort=True, trend=False, priors=PRIORS, num_warmup=1000, num_samples=1000, num_chains=4, seed=0,
        target_accept=0.9, informative=None, dynamic=False, eps=True, alpha_prior='hier', x=None, max_tree_depth=10, rho_err=False):
    d = idx['df']
    tcen = (d['Year'].values - 2024.5).astype(float)   # centred year for trend variant
    kw = dict(u_i=jnp.array(d.u_i.values), q_i=jnp.array(d.q_i.values), t_i=jnp.array(d.t_i.values),
              v_i=jnp.array(d.v_i.values), c_i=jnp.array(d.c_i.values), s=jnp.array(d.s.values.astype(float)),
              tcen=jnp.array(tcen), n_u=idx['n_u'], n_q=idx['n_q'], n_t=idx['n_t'], n_v=idx['n_v'], n_c=idx['n_c'],
              D=jnp.array(d.D.values.astype(float)), cohort=cohort, trend=trend, priors=priors, informative=informative, dynamic=dynamic,
              eps=eps, alpha_prior=alpha_prior, x=(None if x is None else jnp.array(np.asarray(x, dtype=float))), rho_err=rho_err)
    kernel = NUTS(model, target_accept_prob=target_accept, max_tree_depth=max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains,
                chain_method='sequential', progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=('diverging', 'num_steps'), **kw)
    return mcmc, kw


def diagnostics(mcmc, params=('mu', 'sd_alpha', 'sd_gamma', 'sd_u', 'sd_v', 'sd_c', 'sd_eps', 'sd_b', 'sd_eta', 'beta'), ci_prob=0.95):
    import arviz as az
    idata = az.from_numpyro(mcmc)
    present = [p for p in params if p in idata.posterior and idata.posterior[p].ndim > 2 or p in idata.posterior and idata.posterior[p].ndim == 2]
    present = [p for p in params if p in idata.posterior]
    summ = az.summary(idata, var_names=present, ci_prob=ci_prob)
    div = int(idata.sample_stats['diverging'].values.sum())
    vars_ = [v for v in idata.posterior.data_vars if v.endswith('_z') or v.endswith('_free') or v in present]
    ess_b = az.ess(idata, var_names=vars_, method='bulk'); ess_t = az.ess(idata, var_names=vars_, method='tail'); rh = az.rhat(idata, var_names=vars_)
    out = dict(divergences=div, min_ess_bulk=min(float(ess_b[v].min()) for v in ess_b.data_vars),
               min_ess_tail=min(float(ess_t[v].min()) for v in ess_t.data_vars), max_rhat=max(float(rh[v].max()) for v in rh.data_vars))
    try:
        ns = np.asarray(mcmc.get_extra_fields(group_by_chain=True)['num_steps'])
        md = int(np.log2(ns.max())) if ns.max() > 0 else 0
        out['share_iters_at_max_tree_depth'] = float((ns >= 2 ** md).mean()) if md >= 10 else 0.0
        out['max_tree_depth_reached'] = md
    except Exception:
        pass
    return summ, out, idata


def latent_draws(samples, kw, d):
    """Recover latent theta = m + eps for each observation, using the analytic conditional of eps.
    Returns array (n_draws, n_obs)."""
    m = np.asarray(samples['m'])                      # (S, N)
    sd_eps = np.asarray(samples['sd_eps'])[:, None]   # (S,1)
    s = d['s'].values.astype(float)[None, :]
    D = d['D'].values.astype(float)[None, :]
    var_e = sd_eps ** 2
    post_mean = m + (D - m) * var_e / (s ** 2 + var_e)
    post_sd = np.sqrt(var_e * s ** 2 / (s ** 2 + var_e))
    rng = np.random.default_rng(1)
    return post_mean + post_sd * rng.standard_normal(m.shape)


def flatten_samples(mcmc):
    return {k: np.asarray(v).reshape((-1,) + v.shape[2:]) if v.ndim > 1 else np.asarray(v).reshape(-1)
            for k, v in mcmc.get_samples(group_by_chain=True).items()}


# ======================================================================================================
# M4: extended model motivated by posterior predictive checks of the guide-spec model
#   (i) theme-specific loadings on the persistent provider effect (lambda_q, lambda_1 = 1) and theme-specific
#       provider x theme SDs (sigma_v,q): different themes have different between-provider dispersion;
#   (ii) year-specific scaling of the persistent provider effect (kappa_t, kappa_2023 = 1): sector convergence/divergence;
#   (iii) EQUICORRELATED SAMPLING ERROR across the seven themes within a provider-year (rho): the same respondents
#        answer all themes, so OfS sampling errors are not independent across themes.  e_pqt = s_pqt (sqrt(rho) g_pt + sqrt(1-rho) z_pqt).
#   theta_pqt = mu + alpha_q + gamma_t + kappa_t * lambda_q * u_p + v_pq + c_pt      (no eps, no RW: both ~0 in M1/M3)
# ======================================================================================================
def model_v4(u_i, q_i, t_i, v_i, c_i, s, n_u, n_q, n_t, n_v, n_c, D=None, priors=PRIORS, rho_fixed=None, kappa=True):
    p = priors
    mu = numpyro.sample('mu', dist.Normal(0., p['mu_sd']))
    with numpyro.plate('theme_free', n_q - 1):
        alpha_free = numpyro.sample('alpha_free', dist.Normal(0., 3.0))
    alpha = numpyro.deterministic('alpha', jnp.concatenate([alpha_free, -jnp.sum(alpha_free, keepdims=True)]))
    sd_gamma = numpyro.sample('sd_gamma', dist.HalfNormal(p['sd_gamma']))
    with numpyro.plate('year', n_t):
        gamma_z = numpyro.sample('gamma_z', dist.Normal(0., 1.))
    gamma = numpyro.deterministic('gamma', gamma_z * sd_gamma)
    sd_u = numpyro.sample('sd_u', dist.HalfNormal(p['sd_u']))
    with numpyro.plate('unit', n_u):
        u_z = numpyro.sample('u_z', dist.Normal(0., 1.))
    u = numpyro.deterministic('u', u_z * sd_u)
    with numpyro.plate('theme_load', n_q - 1):
        lam_free = numpyro.sample('lam_free', dist.Normal(1., 0.5))
    lam = numpyro.deterministic('lam', jnp.concatenate([jnp.ones(1), lam_free]))
    if kappa:
        with numpyro.plate('year_load', n_t - 1):
            kap_free = numpyro.sample('kap_free', dist.Normal(1., 0.3))
        kap = numpyro.deterministic('kappa', jnp.concatenate([jnp.ones(1), kap_free]))
    else:
        kap = jnp.ones(n_t)
    with numpyro.plate('theme_sdv', n_q):
        sd_v_q = numpyro.sample('sd_v_q', dist.HalfNormal(p['sd_v']))
    with numpyro.plate('unit_theme', n_v):
        v_z = numpyro.sample('v_z', dist.Normal(0., 1.))
    # v_pq scaled by its theme's sd: need theme index of each v -> passed via q_of_v
    sd_c = numpyro.sample('sd_c', dist.HalfNormal(p['sd_c']))
    with numpyro.plate('unit_year', n_c):
        c_z = numpyro.sample('c_z', dist.Normal(0., 1.))
        g = numpyro.sample('g', dist.Normal(0., 1.))          # shared standardised sampling error per provider-year
    c = numpyro.deterministic('c', c_z * sd_c)
    if rho_fixed is None:
        rho = numpyro.sample('rho', dist.Beta(2., 2.))
    else:
        rho = rho_fixed
    v = numpyro.deterministic('v', v_z * sd_v_q[model_v4.q_of_v])
    m = mu + alpha[q_i] + gamma[t_i] + kap[t_i] * lam[q_i] * u[u_i] + v[v_i] + c[c_i]
    numpyro.deterministic('m', m)
    with numpyro.plate('obs', len(s)):
        numpyro.sample('D', dist.Normal(m + s * jnp.sqrt(rho) * g[c_i], s * jnp.sqrt(1. - rho)), obs=D)


def fit_v4(idx, priors=PRIORS, num_warmup=1000, num_samples=1000, num_chains=4, seed=0, target_accept=0.95, rho_fixed=None, kappa=True):
    d = idx['df']
    model_v4.q_of_v = jnp.array(idx['uq'].sort_values('v_i')['q_i'].values)
    kw = dict(u_i=jnp.array(d.u_i.values), q_i=jnp.array(d.q_i.values), t_i=jnp.array(d.t_i.values), v_i=jnp.array(d.v_i.values), c_i=jnp.array(d.c_i.values),
              s=jnp.array(d.s.values.astype(float)), n_u=idx['n_u'], n_q=idx['n_q'], n_t=idx['n_t'], n_v=idx['n_v'], n_c=idx['n_c'],
              D=jnp.array(d.D.values.astype(float)), priors=priors, rho_fixed=rho_fixed, kappa=kappa)
    kernel = NUTS(model_v4, target_accept_prob=target_accept, max_tree_depth=10)
    mcmc = MCMC(kernel, num_warmup=num_warmup, num_samples=num_samples, num_chains=num_chains, chain_method='sequential', progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=('diverging', 'num_steps'), **kw)
    return mcmc, kw
