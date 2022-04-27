//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data
data {
  int<lower=1> n_trials;
  int<lower=1> n_conditions;
  array[n_trials, n_conditions] int choice; // choice can be 0 or 1 
  array[n_trials, n_conditions] int feedback; // feedback can be -1 or 1
  real temp; // tau
}

transformed data {
  vector[2] initValue; //initial values
  initValue = rep_vector(0.0, 2); // initial values = repeat 0 twice
}

// The parameters accepted by the model.
parameters {
  array[n_conditions] real alpha; // learning rate
}

// The model to be estimated.
model {
  real pe; // prediction error
  vector[2] theta; // defining theta (i.e. the rate of choice)
  vector[2] value; 
  
  target += uniform_lpdf(alpha | 0, 1); // prior for learing rate
  
  value = initValue; // define initial value
  
  for (c in 1:n_conditions) {
    for (t in 1:n_trials) {
      
      // calculate the probability of choosing each deck, theta, based on the value 
      theta = softmax(temp * value);
      
      // making choices dependent on theta
      target += categorical_lpmf(choice[t, c] | theta);
      
      // calculate prediction error
      pe = feedback[t, c] - value[choice[t, c]];
        
      // defining value 
      value[choice[t, c]] = value[choice[t, c]] + alpha[c] * pe; // update value
      
    }
  }
}
  
generated quantities { 
  real<lower=0, upper=1> alpha_prior;
  
  real pe;
  vector[2] value;
  vector[2] theta;
  
  real log_lik;
  
  alpha_prior = uniform_rng(0, 1);
  
  value = initValue;
  log_lik = 0;
  
  for (c in 1:n_conditions) {
    for (t in 1:n_trials) {
      
      theta = softmax(temp * value);
      
      log_lik = log_lik + categorical_lpmf(choice[t, c] | theta);
      
      pe = feedback[t, c] - value[choice[t, c]];
        
      value[choice[t, c]] = value[choice[t, c]] + alpha[c] * pe;
    
    }
  }
}

