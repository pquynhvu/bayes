data {
  int<lower = 0> N; // age
  int<lower = 0> M; // year
  matrix[N, M] X; // age
  matrix[N, M] P; // pop
  int y[N, M]; // deaths 
}
parameters {
  vector<lower = 0>[M] alpha; 
  vector<lower = 0>[M] beta;
}
transformed parameters{
  matrix[N, M] eta;
  // matrix[N, M] mu;
  for(j in 1:M){
    //mu[:, j] = alpha[j]*exp(beta[j] * X[:, j]);
    eta[:, j] = beta[j] * X[:, j] + log(alpha[j]*P[:, j]);
  }
}
model {
  for(j in 1:M){  
    target += normal_lpdf(alpha[j] | 0.5, 0.002); 
    target += normal_lpdf(beta[j] | 0.03, 0.0055);
  }
  for(i in 1:N){
    for(j in 1:M){
        target += poisson_log_lpmf(y[i, j] | eta[i, j]);
    }
  }
}

