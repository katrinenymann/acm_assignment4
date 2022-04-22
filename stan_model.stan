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
  array[n_trials, n_conditions] real <lower=-1, upper=1> value1;
  array[n_trials, n_conditions] real<lower=-1, upper=1> value2;
  array[n_trials, n_conditions] int feedback; // feedback can be -1 or 1
  real temp;
}

// The parameters accepted by the model.
parameters {
  array[n_conditions] real alpha; // learning rate
}

// The model to be estimated.
model {
  real pe; // prediction error
  vector[2] theta; // defining theta (i.e. the rate of choice)
  
  target += uniform_lpdf(alpha | 0, 1); // prior for learing rate
  
  for (c in 1:n_conditions) {
    for (t in 1:n_trials) {
      
      // calculate the probability of choosing each deck, theta 
      theta = softmax(temp * [value1[t, c], value2[t, c]]);
      
      // making choices dependent on theta
      target += categorical_lmpf(choice[t, c] | theta);
      
      // calculate prediction error
      pe = feedback[t, c] - value[choice[t, c]];
      
      // defining value 
      value[choice[t, c]] = value[choice[t, c]] + alpha[c] * pe; 
    
    }
  }
}

