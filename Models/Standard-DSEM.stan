data {
  int<lower=1> N; 		 	// number of subjects
  int<lower=1> T; 		 	// number of observations
  array[N] vector[T] Y; 		// time series data
}

parameters {
  vector[3] gamma; 	      	 	// population-level effects
  array[N] vector[3] u;  	 	// subject-specific deviations

  // for covariance matrix of subject-specific deviations
  cholesky_factor_corr[3] L_Omega; 	// Cholesky factor
  vector<lower=0>[3] tau_Omega;		// vector of standard deviations
}

transformed parameters {
  // construct covariance matrix from Cholesky factors and standard deviations
  corr_matrix[3] R_Omega; // correlation matrix
  R_Omega = multiply_lower_tri_self_transpose(L_Omega); // R = L* L'
  // quad_form_diag: diag_matrix(tau) * R * diag_matrix(tau)
  cov_matrix[3] Omega = quad_form_diag(R_Omega, tau_Omega);
  
  // subject-specific parameters
  vector[N] mu;  			// mean
  vector[N] phi; 			// autoregression
  vector[N] psi; 			// residual variance

  // object for deviation from mean
  array[N] vector[T] delta;	

  for (i in 1:N) {
    // to obtain subject-specific effects, 
    // sum population-level effects and subjects-specific deviations
    mu[i] = gamma[1] + u[i][1];
    phi[i] = gamma[2] + u[i][2];
    // note gamma[3] and u[i][3] are estimated on a log-scale
    // to assume normal distribution which simplifies estimation
    psi[i] = exp(gamma[3] + u[i][3]);
    // calculate deviations by subtracting mean
    delta[i] = Y[i] - mu[i];
  }
}

model {
  // prior distributions
  // .. for the population-level parameters
  target += normal_lpdf(gamma | 0, 3); 
  // .. for the Cholesky factor
  target += lkj_corr_cholesky_lpdf(L_Omega | 1.0);
  // .. for the vector of standard deviations
  target += cauchy_lpdf(tau_Omega | 0, 2.5);

  // likelihood
  for (i in 1:N) {
    // subject-specific deviations
    target += multi_normal_lpdf(u[i] | rep_vector(0,3), Omega);
    
    for (t in 2:T) {
      // data given model parameters
      target += normal_lpdf(delta[i][t] | phi[i] * delta[i][t-1], sqrt(psi[i]));
    }
  }
}

generated quantities {
  // obtain posterior predictives (to check model fit)
  array[N] vector[T] delta_pred;
  for (i in 1:N) {
    for (t in 1:T) {
      delta_pred[i][t] = normal_rng(phi[i] * delta[i][t-1], sqrt(psi[i]));
    }
  }
}
