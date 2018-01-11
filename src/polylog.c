#include "polylog.h"

#include "_clt13.h"
#include "utils.h"

#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>

static inline void
mpz_quotient(mpz_t rop, const mpz_t a, const mpz_t b)
{
    mpz_t tmp;
    mpz_init(tmp);
    mpz_mod_near(tmp, a, b);
    mpz_sub(rop, a, tmp);
    mpz_tdiv_q(rop, rop, b);
    mpz_clear(tmp);
}

static inline void
mpz_quotient_2exp(mpz_t rop, const mpz_t a, const size_t b)
{
    mpz_t tmp;
    mpz_init(tmp);
    mpz_mod_near_ui(tmp, a, 1 << b);
    mpz_sub(rop, a, tmp);
    mpz_tdiv_q_2exp(rop, rop, b);
    mpz_clear(tmp);
}

static switch_state_t *
switch_state_new(clt_state_t *s, size_t wordsize, size_t level, bool verbose)
{
    polylog_state_t *pstate = s->pstate;
    switch_state_t *state;
    mpz_t wk, K, wordsize_mpz;
    mpz_t **ss;

    if (level >= pstate->nlevels - 1) {
        fprintf(stderr, "error: level too large (%lu ≥ %lu)\n", level, pstate->nlevels - 1);
        return NULL;
    }

    if ((state = calloc(1, sizeof state[0])) == NULL)
        return NULL;
    state->level = level;
    state->wordsize = wordsize;
    /* k = η_ℓ / log2(wordsize) */
    state->k = (int) ceil(pstate->etas[level] / log2(wordsize));

    if (verbose)
        fprintf(stderr, "  Generating switch state [%lu->%lu]\n", level, level+1);

    mpz_init_set_ui(wordsize_mpz, wordsize);

    /* wk = (wordsize)ᵏ */
    mpz_init(wk);
    mpz_ui_pow_ui(wk, wordsize, state->k);
    /* K = Π^(ℓ) · wk */
    mpz_init(K);
    mpz_mul(K, wk, pstate->x0s[level]);

    state->ys = calloc(pstate->theta, sizeof state->ys[0]);
    for (size_t i = 0; i < pstate->theta; ++i)
        mpz_init(state->ys[i]);
    /* Sample y_{n+1}, ..., y_Θ ∈ [0, K) */
    for (size_t i = s->n; i < pstate->theta; ++i) {
        mpz_urandomm_aes(state->ys[i], s->rngs[0], K);
    }
    ss = calloc(s->n, sizeof ss[0]);
    for (size_t i = 0; i < s->n; ++i) {
        mpz_t tmp, ginv, f;

        mpz_inits(tmp, ginv, f, NULL);

        mpz_invert(ginv, s->gs[i], pstate->ps[level][i]);
        /* Compute fᵢ */
        mpz_mul(f, ginv, pstate->ps[level + 1][i]);
        mpz_quotient(f, f, pstate->ps[level][i]);
        mpz_mul(f, f, s->gs[i]);
        mpz_mod_near(f, f, pstate->ps[level + 1][i]);
        mpz_mod_near(f, f, s->gs[i]);
        mpz_invert(f, f, s->gs[i]);
        mpz_mul(f, f, ginv);

        ss[i] = calloc(pstate->theta, sizeof ss[i][0]);
        for (size_t j = 0; j < s->n; ++j)
            mpz_init_set_ui(ss[i][j], 0);
        for (size_t j = s->n; j < pstate->theta; ++j) {
            mpz_init(ss[i][j]);
            mpz_urandomb_aes(ss[i][j], s->rngs[0], (int) log2(wordsize));
            mpz_mod_near(ss[i][j], ss[i][j], wordsize_mpz);
        }
        mpz_set_ui(ss[i][i], 1);
        for (size_t j = 0; j < pstate->theta; ++j) {
            gmp_printf("%Zd ", ss[i][j]);
        }
        printf("\n");

        /* XXX multiply by z */
        mpz_mod_near(state->ys[i], f, pstate->ps[level][i]);
        mpz_mul(state->ys[i], state->ys[i], K);
        mpz_quotient(state->ys[i], state->ys[i], pstate->ps[level][i]);

        for (size_t t = s->n; t < pstate->theta; ++t) {
            mpz_mul(tmp, state->ys[t], ss[i][t]);
            mpz_sub(state->ys[i], state->ys[i], tmp);
        }
        mpz_mod_near(state->ys[i], state->ys[i], K);
        mpz_clears(tmp, ginv, f, NULL);
    }

    state->sigmas = calloc(pstate->theta, sizeof state->sigmas[0]);
    for (size_t t = 0; t < pstate->theta; ++t) {
        state->sigmas[t] = calloc(state->k, sizeof state->sigmas[t][0]);
        for (size_t j = 0; j < state->k; ++j) {
            mpz_t *xs = calloc(s->n, sizeof xs[0]);
            state->sigmas[t][j] = clt_elem_new();
            for (size_t i = 0; i < s->n; ++i) {
                mpz_init(xs[i]);
                mpz_ui_pow_ui(xs[i], wordsize, j);
                mpz_mul(xs[i], xs[i], ss[i][t]);
                mpz_mul(xs[i], xs[i], pstate->ps[level + 1][i]);
                mpz_quotient(xs[i], xs[i], wk);
                mpz_mul(xs[i], xs[i], s->gs[i]);
            }
            polylog_encode(state->sigmas[t][j], s, s->n, xs, NULL, level + 1);
            for (size_t i = 0; i < s->n; ++i) {
                mpz_clear(xs[i]);
            }
            free(xs);
        }
    }
    for (size_t i = 0; i < s->n; ++i) {
        for (size_t t = 0; t < pstate->theta; ++t) {
            mpz_clear(ss[i][t]);
        }
        free(ss[i]);
    }
    free(ss);
    mpz_clears(wk, K, wordsize_mpz, NULL);
    return state;
}

void
polylog_state_free(polylog_state_t *state)
{
    if (state == NULL)
        return;
    if (state->etas)
        free(state->etas);
    for (size_t i = 0; i < state->nlevels; ++i) {
        for (size_t j = 0; j < state->n; ++j) {
            mpz_clear(state->ps[i][j]);
        }
        free(state->ps[i]);
    }
    free(state->ps);
    free(state);
}

polylog_state_t *
polylog_state_new(clt_state_t *s, size_t eta, size_t b, size_t wordsize,
                  size_t nlevels, size_t *levels, size_t nswitches)
{
    const bool verbose = s->flags & CLT_FLAG_VERBOSE;
    polylog_state_t *state;
    int count=0;

    if (log2(wordsize) != floor(log2(wordsize))) {
        fprintf(stderr, "error: wordsize must be a power of two\n");
        return NULL;
    }

    if ((state = calloc(1, sizeof state[0])) == NULL)
        return NULL;
    s->pstate = state;
    state->n = s->n;
    state->b = b;               /* XXX probably should generate within this function rather than take as arg */
    state->nlevels = nlevels + 1;
    state->theta = 10;          /* XXX */
    assert(state->theta > state->n);
    if (verbose) {
        fprintf(stderr, "Polylog CLT initialization:\n");
        fprintf(stderr, "  n: ..... %lu\n", state->n);
        fprintf(stderr, "  b: ..... %lu\n", state->b);
        fprintf(stderr, "  nlevels: [0, %lu]\n", state->nlevels - 1);
        fprintf(stderr, "  ηs: .... ");
    }
    state->etas = calloc(state->nlevels, sizeof state->etas[0]);
    for (size_t i = 0; i < state->nlevels; ++i) {
        if (i * 2 * b > eta) {
            fprintf(stderr, "error: η - ℓ·2B < 0\n");
            goto error;
        }
        state->etas[i] = eta - i * 2 * b;
        if (verbose)
            fprintf(stderr, " %lu", state->etas[i]);
    }
    if (verbose) {
        fprintf(stderr, "\n  Generating p_i's:\n");
    }
    state->ps = calloc(state->nlevels, sizeof state->ps[0]);
    for (size_t i = 0; i < state->nlevels; ++i) {
        state->ps[i] = calloc(state->n, sizeof state->ps[i][0]);
        if (verbose) {
            count = 0;
            print_progress(count, state->n);
        }
        for (size_t j = 0; j < state->n; ++j) {
            mpz_init(state->ps[i][j]);
            mpz_prime(state->ps[i][j], s->rngs[j], state->etas[i]);
            if (verbose)
                print_progress(++count, state->n);
        }
        if (verbose)
            fprintf(stderr, "\n");
    }
    if (verbose) {
        fprintf(stderr, "  Generating x0s:\n");
    }
    state->x0s = calloc(state->nlevels, sizeof state->x0s[0]);
    state->crt_coeffs = calloc(state->nlevels, sizeof state->crt_coeffs[0]);
    for (size_t i = 0; i < state->nlevels; ++i) {
        mpz_init(state->x0s[i]);
        if (verbose)
            fprintf(stderr, "  Computing product:\n");
        product(state->x0s[i], state->ps[i], state->n, verbose);
        state->crt_coeffs[i] = mpz_vector_new(state->n);
        crt_coeffs(state->crt_coeffs[i], state->ps[i], state->n, state->x0s[i], verbose);
    }
    state->switches = calloc(nswitches, sizeof state->switches[0]);
    for (size_t i = 0; i < nswitches; ++i) {
        if ((state->switches[i] = switch_state_new(s, wordsize, levels[i], verbose)) == NULL)
            goto error;
    }
    if (verbose)
        fprintf(stderr, "Polylog CLT initialization complete!\n");
    return state;
error:
    polylog_state_free(state);
    return NULL;
}

int
polylog_encode(clt_elem_t *rop, const clt_state_t *s, size_t n, mpz_t *xs, const int *ix, size_t level)
{
    (void) ix;                  /* XXX */
    polylog_state_t *pstate = s->pstate;
    mpz_t b_mpz;
    mpz_init_set_ui(b_mpz, pstate->b);
    mpz_set_ui(rop->elem, 0);
    for (size_t i = 0; i < s->n; ++i) {
        mpz_t tmp;
        mpz_init(tmp);
        /* rᵢ ∈ [-2ᴮ, 2ᴮ ] */
        mpz_urandomb_aes(tmp, s->rngs[i], 2 * pstate->b);
        mpz_mod_near(tmp, tmp, b_mpz);
        /* gᵢ · rᵢ */
        mpz_mul(tmp, tmp, s->gs[i]);
        /* mᵢ + gᵢ·rᵢ */
        mpz_add(tmp, tmp, xs[slot(i,n, s->n)]);
        mpz_mul(tmp, tmp, pstate->crt_coeffs[level][i]);
        mpz_add(rop->elem, rop->elem, tmp);
        mpz_clear(tmp);
    }
    rop->level = level;
    /* XXX ignore index set for now */
    /* if (ix) { */
    /*     mpz_t tmp; */
    /*     mpz_init(tmp); */
    /*     /\* multiply by appropriate zinvs *\/ */
    /*     rop->ix = calloc(s->nzs, sizeof rop->ix[0]); */
    /*     for (unsigned long i = 0; i < s->nzs; ++i) { */
    /*         if (ix[i] <= 0) */
    /*             continue; */
    /*         rop->ix[i] = ix[i]; */
    /*         mpz_powm_ui(tmp, s->zinvs[i], ix[i], pstate->x0s[level]); */
    /*         mpz_mul_mod(rop->elem, rop->elem, tmp, pstate->x0s[level]); */
    /*     } */
    /*     mpz_clear(tmp); */
    /* } */
    mpz_clear(b_mpz);
    return CLT_OK;
}

int
polylog_elem_add(clt_elem_t *rop, const clt_state_t *s, const clt_elem_t *a, const clt_elem_t *b)
{
    if (a->level != b->level) {
        fprintf(stderr, "error: levels unequal (%lu ≠ %lu), unable to add\n",
                a->level, b->level);
        return CLT_ERR;
    }
    mpz_add(rop->elem, a->elem, b->elem);
    mpz_mod(rop->elem, rop->elem, s->pstate->x0s[a->level]);
    rop->level = a->level;
    return CLT_OK;
}

int
polylog_elem_mul(clt_elem_t *rop, const clt_state_t *s, const clt_elem_t *a, const clt_elem_t *b, size_t idx)
{
    switch_state_t *sstate;
    if (a->level != b->level) {
        fprintf(stderr, "error: levels unequal (%lu ≠ %lu), unable to multiply\n",
                a->level, b->level);
        return CLT_ERR;
    }
    sstate = s->pstate->switches[idx];
    if (sstate->level != a->level) {
        fprintf(stderr, "error: switch and element levels unequal (%lu ≠ %lu)\n",
                sstate->level, a->level);
        return CLT_ERR;
    }
    /* a · b mod Π^(ℓ) */
    mpz_mul(rop->elem, a->elem, b->elem);
    mpz_mod(rop->elem, rop->elem, s->pstate->x0s[sstate->level]);
    rop->level = a->level;
    if (polylog_switch(rop, s, rop, sstate, s->flags & CLT_FLAG_VERBOSE) == CLT_ERR)
        return CLT_ERR;
    return CLT_OK;
}

int
polylog_switch(clt_elem_t *rop, const clt_state_t *s, const clt_elem_t *x_, const switch_state_t *sstate, bool verbose)
{
    const polylog_state_t *pstate = s->pstate;
    const double start = current_time();
    mpz_t *pi, *pip, ct, wk;
    clt_elem_t *x;
    int ret = CLT_ERR;

    if (rop == NULL)
        return CLT_ERR;
    mpz_inits(ct, wk, NULL);
    x = clt_elem_new();
    clt_elem_set(x, x_);
    if (sstate->level != x->level) {
        fprintf(stderr, "error: using switch parameter with a mismatched level\n");
        fprintf(stderr, "       element level: %lu\n", x->level);
        fprintf(stderr, "       switch level:  %lu\n", sstate->level);
        goto cleanup;
    }

    pi = &pstate->x0s[sstate->level];
    pip = &pstate->x0s[sstate->level + 1];

    mpz_ui_pow_ui(wk, sstate->wordsize, sstate->k);
    mpz_set_ui(rop->elem, 0);
    for (size_t t = 0; t < pstate->theta; ++t) {
        /* Compute cₜ = (x · yₜ) / Π^(ℓ) mod (wordsize)ᵏ */
        mpz_mul(ct, x->elem, sstate->ys[t]);
        mpz_quotient(ct, ct, *pi);
        mpz_mod(ct, ct, wk);
        for (size_t i = 0; i < sstate->k; ++i) {
            mpz_t tmp, decomp;
            mpz_inits(tmp, decomp, NULL);
            /* Compute word decomposition c_{t,i} of cₜ */
            mpz_mod_near_ui(decomp, ct, sstate->wordsize);
            mpz_quotient_2exp(ct, ct, (int) log2(sstate->wordsize));
            /* σ_{t,i} · c_{t,i} */
            mpz_mul(tmp, decomp, sstate->sigmas[t][i]->elem);
            mpz_add(rop->elem, rop->elem, tmp);
            mpz_mod(rop->elem, rop->elem, *pip);
            mpz_clears(tmp, decomp, NULL);
        }
    }
    rop->level = x->level + 1;
    if (verbose)
        fprintf(stderr, "Switch time: %.2fs\n", current_time() - start);
    ret = CLT_OK;
cleanup:
    mpz_clears(ct, wk, NULL);
    clt_elem_free(x);
    return ret;
}

int
polylog_elem_decrypt(clt_elem_t *x, const clt_state_t *s, size_t level)
{
    size_t nbits=0;
    mpz_t rop;
    mpz_init(rop);
    printf("DECRYPTION @ LEVEL %lu :: ", level);
    for (size_t i = 0; i < s->n; ++i) {
        mpz_mod_near(rop, x->elem, s->pstate->ps[level][i]);
        nbits = mpz_sizeinbase(x->elem, 2);
        mpz_mod_near(rop, rop, s->gs[i]);
        gmp_printf("%Zd ", rop);
    }
    mpz_clear(rop);
    printf(": %lu\n", nbits);
    return CLT_OK;
}
